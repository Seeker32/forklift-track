from __future__ import annotations

from pathlib import Path
from typing import Any


class DebugVideoWriter:
    def __init__(
        self,
        *,
        camera_config: dict[str, Any],
        capture: Any,
        first_frame: Any,
        output_dir: str | Path = "outputs",
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.camera_config = camera_config
        self.output_path = Path(output_dir) / f"debug_{_safe_filename(camera_config['camera_id'])}.mp4"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = first_frame.shape[:2]
        fps = _capture_float(capture, cv2.CAP_PROP_FPS, default=30.0)
        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (int(width), int(height)))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open debug video writer: {self.output_path}")

    def write(
        self,
        *,
        frame: Any,
        frame_index: int,
        detections: list[dict[str, Any]],
        tracks: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> None:
        annotated = frame.copy()
        self._draw_line_zone(annotated)
        self._draw_detections(annotated, detections)
        self._draw_tracks(annotated, tracks)
        self._draw_events(annotated, events)
        self._draw_header(annotated, frame_index, len(detections), len(tracks))
        self.writer.write(annotated)

    def release(self) -> None:
        self.writer.release()

    def _draw_line_zone(self, frame: Any) -> None:
        import numpy as np

        cv2 = self.cv2
        start = self.camera_config["line_start"]
        end = self.camera_config["line_end"]
        width = float(self.camera_config["line_width"])
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0 and width > 0:
            nx = -dy / length * width / 2.0
            ny = dx / length * width / 2.0
            polygon = np.array(
                [
                    [round(x1 + nx), round(y1 + ny)],
                    [round(x2 + nx), round(y2 + ny)],
                    [round(x2 - nx), round(y2 - ny)],
                    [round(x1 - nx), round(y1 - ny)],
                ],
                dtype=np.int32,
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 180, 255))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        cv2.line(frame, _point(start), _point(end), (0, 0, 255), 2)

    def _draw_detections(self, frame: Any, detections: list[dict[str, Any]]) -> None:
        cv2 = self.cv2
        for detection in detections:
            x1, y1, x2, y2 = _bbox(detection.get("bbox", []))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 1)
            label = f"{detection.get('class_name', '?')} {float(detection.get('score', 0.0)):.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    def _draw_tracks(self, frame: Any, tracks: list[dict[str, Any]]) -> None:
        cv2 = self.cv2
        for track in tracks:
            x1, y1, x2, y2 = _bbox(track.get("bbox", []))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            point = _representative_point(track)
            cv2.circle(frame, point, 4, (0, 255, 255), -1)
            label = f"id {int(track['track_id'])}"
            cv2.putText(frame, label, (x1, max(15, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    def _draw_events(self, frame: Any, events: list[dict[str, Any]]) -> None:
        cv2 = self.cv2
        for index, event in enumerate(events):
            label = f"event {event['direction']} id {event['track_id']}"
            cv2.putText(frame, label, (12, 54 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_header(self, frame: Any, frame_index: int, detection_count: int, track_count: int) -> None:
        cv2 = self.cv2
        label = (
            f"{self.camera_config['camera_id']} frame={frame_index} "
            f"detections={detection_count} tracks={track_count}"
        )
        cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)


def _safe_filename(value: Any) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def _capture_float(capture: Any, prop: int, *, default: float) -> float:
    getter = getattr(capture, "get", None)
    if getter is None:
        return default
    try:
        return float(getter(prop))
    except (TypeError, ValueError):
        return default


def _bbox(value: Any) -> tuple[int, int, int, int]:
    if not value or len(value) < 4:
        return 0, 0, 0, 0
    return tuple(round(float(coordinate)) for coordinate in value[:4])


def _point(value: Any) -> tuple[int, int]:
    return round(float(value[0])), round(float(value[1]))


def _representative_point(track: dict[str, Any]) -> tuple[int, int]:
    bbox = track.get("bbox")
    if bbox and len(bbox) >= 4:
        return round((float(bbox[0]) + float(bbox[2])) / 2.0), round(float(bbox[3]))
    return _point(track["center"])
