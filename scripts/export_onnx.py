"""Convert RF-DETR Small fine-tuned checkpoint to ONNX.

Usage: uv run python scripts/export_onnx.py
Output: models/inference_model.onnx
"""

import argparse
from pathlib import Path

import torch

from rfdetr.config import RFDETRSmallConfig
from rfdetr.main import populate_args

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_args(checkpoint: dict) -> argparse.Namespace:
    """Build model args from checkpoint metadata and RFDETRSmallConfig defaults."""
    ckpt_args = checkpoint["args"]
    config = RFDETRSmallConfig()

    return populate_args(
        num_classes=2,
        resolution=512,
        pretrain_weights=None,
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
        dec_layers=config.dec_layers,
        sa_nheads=config.sa_nheads,
        ca_nheads=config.ca_nheads,
        dec_n_points=config.dec_n_points,
        num_queries=config.num_queries,
        num_select=config.num_select,
        projector_scale=config.projector_scale,
        out_feature_indexes=config.out_feature_indexes,
        bbox_reparam=config.bbox_reparam,
        lite_refpoint_refine=config.lite_refpoint_refine,
        two_stage=config.two_stage,
        group_detr=ckpt_args.get("group_detr", 13),
        position_embedding="sine",
        use_cls_token=False,
        layer_norm=config.layer_norm,
        positional_encoding_size=config.positional_encoding_size,
        device="cpu",
        aux_loss=False,
        segmentation_head=False,
    )


def main() -> None:
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoint_best_regular.pth"
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )

    class_names = checkpoint["args"].get(
        "class_names", ["forklift_with_load", "forklift_empty"]
    )
    print(f"Classes: {class_names}")

    # Build model
    print("Building model...")
    from rfdetr.models.lwdetr import build_model

    args = build_args(checkpoint)
    model = build_model(args)

    # Load fine-tuned weights
    print("Loading fine-tuned weights...")
    state_dict = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Switch to export-compatible forward pass
    model.export()
    model.eval()

    # Prepare dummy input
    print("Exporting to ONNX...")
    dummy = torch.randn(1, 3, 512, 512)

    output_path = output_dir / "inference_model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["dets", "labels"],
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved: {output_path}")


if __name__ == "__main__":
    main()
