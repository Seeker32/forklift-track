from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

_CLASS_NAME_TO_ID: dict[str, int] = {"forklift_with_load": 0, "forklift_empty": 1}
_CLASS_ID_TO_NAME: dict[int, str] = {0: "forklift_with_load", 1: "forklift_empty"}


class ByteTrackTracker:
    """ByteTrack adapter with the project's stable tracking output format."""

    def __init__(
        self,
        frame_rate: int = 30,
        track_buffer: int = 30,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        tracker_backend: Any | None = None,
    ) -> None:
        self.tracker_backend = tracker_backend or self._create_tracker_backend(
            frame_rate=frame_rate,
            track_buffer=track_buffer,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
        )

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tracked_rows = self.tracker_backend.update(_DetectionResults.from_detections(detections))
        return [self._track_from_row(row) for row in tracked_rows]

    @staticmethod
    def _create_tracker_backend(
        frame_rate: int,
        track_buffer: int,
        track_high_thresh: float,
        track_low_thresh: float,
        new_track_thresh: float,
        match_thresh: float,
    ) -> Any:
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_buffer=track_buffer,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        return BYTETracker(args, frame_rate=frame_rate)

    @staticmethod
    def _track_from_row(row: Any) -> dict[str, Any]:
        values = [float(value) for value in row]
        x1, y1, x2, y2 = values[:4]
        class_id = int(values[6]) if len(values) > 6 else 0
        return {
            "track_id": int(values[4]),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "score": float(values[5]),
            "class_name": _CLASS_ID_TO_NAME.get(class_id, "forklift_with_load"),
        }


class _DetectionResults:
    def __init__(self, boxes: np.ndarray) -> None:
        boxes = np.atleast_2d(boxes).astype(np.float32, copy=False)
        self.xyxy = boxes[:, :4]
        self.conf = boxes[:, 4]
        self.cls = boxes[:, 5]

    @classmethod
    def from_detections(cls, detections: list[dict[str, Any]]) -> "_DetectionResults":
        rows = [
            [
                float(detection["bbox"][0]),
                float(detection["bbox"][1]),
                float(detection["bbox"][2]),
                float(detection["bbox"][3]),
                float(detection["score"]),
                float(_CLASS_NAME_TO_ID.get(detection.get("class_name", ""), 0)),
            ]
            for detection in detections
        ]
        if not rows:
            return cls(np.empty((0, 6), dtype=np.float32))
        return cls(np.asarray(rows, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index: Any) -> "_DetectionResults":
        return self.__class__(self._boxes[index])

    @property
    def _boxes(self) -> np.ndarray:
        return np.column_stack((self.xyxy, self.conf, self.cls)).astype(np.float32, copy=False)

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2
        return xywh
