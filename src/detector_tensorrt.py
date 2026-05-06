"""Compatibility shim for removed TensorRT .engine support."""

from __future__ import annotations

class TensorRTForkliftDetector:
    """Backward-compatible error for deprecated TensorRT .engine usage."""

    def __init__(
        self,
        engine_path: str,
        confidence: float = 0.4,
        class_names: list[str] | None = None,
        allowed_class_names: list[str] | None = None,
        runtime_loader: object | None = None,
    ) -> None:
        raise RuntimeError(
            "TensorRT .engine models are no longer supported. Deploy the RF-DETR model as .onnx and use "
            "ONNX Runtime with the TensorRT Execution Provider instead."
        )
