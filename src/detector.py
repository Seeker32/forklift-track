from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any
import warnings

ModelLoader = Callable[[str], Any]


class ForkliftDetector:
    """YOLO-based forklift detector with a stable output format."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        class_name: str = "forklift_2",
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.class_name = class_name
        self.model = (model_loader or self._load_ultralytics_model)(model_path)

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            if hasattr(self.model, "conf"):
                self.model.conf = self.confidence
            with self._suppress_legacy_yolov5_autocast_warning():
                results = self.model(frame, size=640)

        if hasattr(results, "xyxy"):
            return self._detections_from_yolov5_results(results)

        detections: list[dict[str, Any]] = []

        for result in results:
            names = getattr(result, "names", getattr(self.model, "names", {}))
            for box in getattr(result, "boxes", []):
                class_id = int(self._scalar(box.cls))
                detected_class = names[class_id] if isinstance(names, dict) else names[class_id]
                if detected_class != self.class_name:
                    continue

                detections.append(
                    {
                        "bbox": [float(value) for value in self._values(box.xyxy[0])],
                        "score": float(self._scalar(box.conf)),
                        "class_name": detected_class,
                    }
                )

        return detections

    @staticmethod
    @contextmanager
    def _suppress_legacy_yolov5_autocast_warning() -> Iterator[None]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\.",
                category=FutureWarning,
            )
            yield

    def _detections_from_yolov5_results(self, results: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        names = getattr(results, "names", getattr(self.model, "names", {}))

        for row in getattr(results, "xyxy", [[]])[0]:
            values = self._values(row)
            if len(values) < 6:
                continue

            class_id = int(values[5])
            detected_class = names[class_id] if isinstance(names, dict) else names[class_id]
            if detected_class != self.class_name:
                continue

            score = float(values[4])
            if score < self.confidence:
                continue

            detections.append(
                {
                    "bbox": [float(value) for value in values[:4]],
                    "score": score,
                    "class_name": detected_class,
                }
            )

        return detections

    @staticmethod
    def _load_ultralytics_model(model_path: str) -> Any:
        if ForkliftDetector._requires_legacy_yolov5(model_path):
            return ForkliftDetector._load_legacy_yolov5_model(model_path)

        from ultralytics import YOLO

        try:
            return YOLO(model_path)
        except Exception as exc:
            if "models.yolo" not in str(exc) and "Weights only load failed" not in str(exc):
                raise
            return ForkliftDetector._load_legacy_yolov5_model(model_path)

    @staticmethod
    def _requires_legacy_yolov5(model_path: str) -> bool:
        try:
            with open(model_path, "rb") as weight_file:
                return b"models.yolo" in weight_file.read(131072)
        except OSError:
            return False

    @staticmethod
    def _load_legacy_yolov5_model(model_path: str) -> Any:
        import torch
        import yolov5

        yolov5_root = str(Path(yolov5.__file__).resolve().parent)
        if yolov5_root not in sys.path:
            sys.path.insert(0, yolov5_root)
        sys.modules.pop("models", None)

        original_load = torch.load

        def load_trusted_checkpoint(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = load_trusted_checkpoint
        try:
            model = yolov5.load(str(Path(model_path).resolve()), device="cpu")
        finally:
            torch.load = original_load

        return model

    @classmethod
    def _scalar(cls, value: Any) -> float:
        values = cls._values(value)
        if not values:
            raise ValueError("Expected scalar value, got empty sequence")
        return float(values[0])

    @staticmethod
    def _values(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
