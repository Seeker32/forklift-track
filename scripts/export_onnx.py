import os

import torch
import torch.nn.functional as F
from rfdetr import RFDETRSmall
from rfdetr.utilities import box_ops

# Monkey-patch: ONNX doesn't support _upsample_bicubic2d_aa (antialias=True).
# RFDETR uses this only for positional embedding interpolation, where
# anti-aliasing is irrelevant — safe to disable.
_orig_interpolate = F.interpolate


def _interpolate_no_aa(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    return _orig_interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=False,
    )


F.interpolate = _interpolate_no_aa

CHECKPOINT_PATH = "../models/checkpoint_best_total.pth"
OUTPUT_PATH = "../models/model.onnx"
RESOLUTION = 512
NUM_CLASSES = 2
NUM_SELECT = 300


class DeployExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_classes: int, num_select: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_select = num_select

    def forward(self, x: torch.Tensor, target_sizes: torch.Tensor):
        outputs = self.model(x)
        pred_logits = outputs["pred_logits"][..., : self.num_classes]
        pred_boxes = outputs["pred_boxes"]

        probs = pred_logits.sigmoid()
        batch_size, num_queries, num_classes = probs.shape

        topk_values, topk_indexes = torch.topk(
            probs.reshape(batch_size, num_queries * num_classes),
            k=min(self.num_select, num_queries * num_classes),
            dim=1,
        )
        topk_boxes = torch.div(topk_indexes, num_classes, rounding_mode="floor")
        labels = topk_indexes % num_classes

        boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.dtype)
        boxes = boxes * scale_fct[:, None, :]

        return topk_values, labels.to(torch.int64), boxes


print(f"Loading checkpoint: {CHECKPOINT_PATH}")
model = RFDETRSmall(pretrain_weights=CHECKPOINT_PATH, num_classes=NUM_CLASSES, device="cuda:0")
raw_model = model.model.model
raw_model.eval()

wrapper = DeployExportWrapper(raw_model, num_classes=NUM_CLASSES, num_select=NUM_SELECT)
dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)
dummy_target_sizes = torch.tensor([[RESOLUTION, RESOLUTION]], dtype=torch.int64)

print(f"Exporting deployable ONNX to: {OUTPUT_PATH}")
torch.onnx.export(
    wrapper,
    (dummy_input, dummy_target_sizes),
    OUTPUT_PATH,
    input_names=["input", "target_sizes"],
    output_names=["scores", "labels", "boxes"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "target_sizes": {0: "batch_size"},
        "scores": {0: "batch_size"},
        "labels": {0: "batch_size"},
        "boxes": {0: "batch_size"},
    },
    opset_version=17,
    dynamo=False,
)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Done. ONNX model saved ({size_mb:.1f} MB)")
