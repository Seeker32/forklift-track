"""ONNX Runtime based forklift detector for RF-DETR models."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort


class ONNXForkliftDetector:
    """RF-DETR forklift detector backed by ONNX Runtime.

    Input resolution: 512x512.
    Output classes: forklift_with_load, forklift_empty.
    """

    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str,
        confidence: float = 0.4,
        class_names: list[str] | None = None,
        allowed_class_names: list[str] | None = None,
        num_select: int = 300,
    ) -> None:
        self.confidence = confidence
        self.class_names = class_names or ["forklift_with_load", "forklift_empty"]
        self.allowed_class_names = allowed_class_names
        self.num_select = num_select
        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        output_names = {o.name for o in self._session.get_outputs()}
        self._is_deploy_format = {"scores", "labels", "boxes"}.issubset(output_names)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        if self._is_deploy_format:
            scores, labels, boxes = self._session.run(
                ["scores", "labels", "boxes"],
                {
                    "input": input_tensor,
                    "target_sizes": np.array([[h, w]], dtype=np.int64),
                },
            )
            detections = self._postprocess_deploy(scores, labels, boxes, h, w)
        else:
            dets, labels = self._session.run(None, {"input": input_tensor})
            detections = self._postprocess(dets, labels, h, w)

        if self.allowed_class_names is not None:
            detections = [d for d in detections if d.get("class_name") in self.allowed_class_names]
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.MEANS) / self.STDS
        return np.transpose(image, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def _postprocess(
        self, dets: np.ndarray, labels: np.ndarray, orig_h: int, orig_w: int
    ) -> list[dict[str, Any]]:
        # dets: [1, num_queries, 4] — cxcywh normalized to [0,1]
        # labels: [1, num_queries, num_classes] — raw logits
        boxes = dets[0]
        class_logits = labels[0]
        num_classes = len(self.class_names)

        if class_logits.shape[1] < num_classes:
            raise RuntimeError(
                f"Model returned {class_logits.shape[1]} class logits, expected at least {num_classes}."
            )

        # RF-DETR library export returns raw logits; keep only the application classes.
        probs = 1.0 / (1.0 + np.exp(-class_logits[:, :num_classes]))

        flat_probs = probs.reshape(-1)
        num_select = min(self.num_select, flat_probs.shape[0])
        topk_indices = np.argpartition(-flat_probs, num_select - 1)[:num_select]
        topk_scores = flat_probs[topk_indices]
        order = np.argsort(-topk_scores)
        topk_indices = topk_indices[order]
        topk_scores = topk_scores[order]

        box_indices = topk_indices // num_classes
        class_ids = topk_indices % num_classes

        selected_boxes = boxes[box_indices]
        cx, cy, bw, bh = selected_boxes[:, 0], selected_boxes[:, 1], selected_boxes[:, 2], selected_boxes[:, 3]
        x1 = np.clip((cx - bw / 2.0) * orig_w, 0, orig_w - 1)
        y1 = np.clip((cy - bh / 2.0) * orig_h, 0, orig_h - 1)
        x2 = np.clip((cx + bw / 2.0) * orig_w, 0, orig_w - 1)
        y2 = np.clip((cy + bh / 2.0) * orig_h, 0, orig_h - 1)

        detections: list[dict[str, Any]] = []
        for i in range(len(topk_scores)):
            score = topk_scores[i]
            if score < self.confidence:
                continue
            cls_id = int(class_ids[i])
            detections.append(
                {
                    "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    "score": float(score),
                    "class_name": self.class_names[cls_id],
                }
            )

        return detections

    def _postprocess_deploy(
        self, scores: np.ndarray, labels: np.ndarray, boxes: np.ndarray,
        orig_h: int, orig_w: int,
    ) -> list[dict[str, Any]]:
        batch_scores = scores[0]
        batch_labels = labels[0]
        batch_boxes = boxes[0]

        detections: list[dict[str, Any]] = []
        for i in range(len(batch_scores)):
            score = float(batch_scores[i])
            if score < self.confidence:
                continue
            cls = int(batch_labels[i])
            if cls < 0 or cls >= len(self.class_names):
                continue
            box = batch_boxes[i]
            detections.append(
                {
                    "bbox": [
                        float(np.clip(box[0], 0, orig_w - 1)),
                        float(np.clip(box[1], 0, orig_h - 1)),
                        float(np.clip(box[2], 0, orig_w - 1)),
                        float(np.clip(box[3], 0, orig_h - 1)),
                    ],
                    "score": score,
                    "class_name": self.class_names[cls],
                }
            )

        return detections
