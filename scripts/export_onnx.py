from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoint_best_regular.pth"
OUTPUT_PATH = PROJECT_ROOT / "models" / "inference_model.onnx"
RESOLUTION = 512


def build_args(args_dict: dict[str, Any]) -> Namespace:
    return Namespace(**args_dict)


def load_model(checkpoint_path: Path) -> Any:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    from rfdetr.models.lwdetr import build_model

    defaults = {
        "num_classes": 2,
        "device": "cpu",
        "encoder": "dinov2_windowed_small",
        "vit_encoder_num_layers": 12,
        "pretrained_encoder": None,
        "window_block_indexes": None,
        "drop_path": 0.0,
        "dropout": 0.0,
        "hidden_dim": 256,
        "dim_feedforward": 2048,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "use_cls_token": False,
        "position_embedding": "sine",
        "freeze_encoder": False,
        "layer_norm": True,
        "rms_norm": False,
        "backbone_lora": False,
        "force_no_pretrain": False,
        "gradient_checkpointing": False,
        "pretrain_weights": str(checkpoint_path),
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "resolution": RESOLUTION,
        "shape": (RESOLUTION, RESOLUTION),
        "encoder_only": False,
        "backbone_only": False,
        "dec_layers": 3,
        "dec_n_points": 2,
        "decoder_norm": "LN",
        "mask_downsample_ratio": 4,
        "segmentation_head": False,
        "num_queries": 300,
        "aux_loss": True,
        "group_detr": 13,
        "two_stage": True,
        "lite_refpoint_refine": True,
        "bbox_reparam": True,
        "cls_loss_coef": 1.0,
        "bbox_loss_coef": 5,
        "num_select": 300,
        "dataset_file": "roboflow",
        "ia_bce_loss": True,
        "square_resize_div_64": True,
    }
    args = build_args(defaults)
    for k, v in checkpoint.get("args", {}).items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)
    args.device = "cpu"
    model = build_model(args)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def _patch_interpolate_no_aa() -> Any:
    original_interpolate = F.interpolate

    def _interpolate_no_aa(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        return original_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=False,
        )

    F.interpolate = _interpolate_no_aa
    return original_interpolate


def _patch_projector_layer_norm_for_onnx() -> Any:
    projector_module = sys.modules.get("rfdetr.models.backbone.projector")
    if projector_module is None:
        return None, None
    LayerNorm = projector_module.LayerNorm

    original_forward = LayerNorm.forward

    def _forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

    LayerNorm.forward = _forward
    return LayerNorm, original_forward


def main() -> None:
    original_interpolate = _patch_interpolate_no_aa()
    layer_norm_cls = None
    original_layer_norm_forward = None
    try:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        model = load_model(CHECKPOINT_PATH)
        layer_norm_cls, original_layer_norm_forward = _patch_projector_layer_norm_for_onnx()
        if hasattr(model, "export"):
            model.export()
        dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)

        print(f"Exporting raw ONNX to: {OUTPUT_PATH}")
        torch.onnx.export(
            model,
            dummy_input,
            str(OUTPUT_PATH),
            input_names=["input"],
            output_names=["pred_boxes", "pred_logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
                "pred_logits": {0: "batch_size"},
            },
            opset_version=18,
            dynamo=False,
        )
    finally:
        F.interpolate = original_interpolate
        if layer_norm_cls is not None and original_layer_norm_forward is not None:
            layer_norm_cls.forward = original_layer_norm_forward

    try:
        import onnx

        if OUTPUT_PATH.exists():
            onnx_model = onnx.load(str(OUTPUT_PATH))
            onnx.checker.check_model(onnx_model)
    except ModuleNotFoundError:
        pass

    if OUTPUT_PATH.exists():
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"Done. ONNX model saved ({size_mb:.1f} MB)")
    else:
        print("Done. ONNX export completed.")


if __name__ == "__main__":
    main()
