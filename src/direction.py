from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.events import create_event
from src.geometry import crossed_line, dot_product, movement_vector


@dataclass
class DirectionDetector:
    camera_id: str
    line_start: Sequence[float]
    line_end: Sequence[float]
    in_direction: Sequence[float]
    max_missing_frames: int = 30
    line_width: float = 40.0
    last_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    last_seen_frames: dict[int, int] = field(default_factory=dict)
    crossed_active: dict[int, bool] = field(default_factory=dict)
    current_frame: int = 0
    last_debug_snapshot: dict[str, Any] | None = None

    def update(self, track: dict[str, Any], frame_index: int | None = None) -> dict[str, Any] | None:
        self._advance_frame(frame_index)
        track_id = int(track["track_id"])
        center, bottom = self._representative_points(track)
        self.last_seen_frames[track_id] = self.current_frame
        self.prune_missing(self.current_frame)

        previous = self.last_centers.get(track_id)
        self.last_centers[track_id] = bottom

        crossed = crossed_line(center, bottom, self.line_start, self.line_end)

        # Once the bbox clears the line, unlock so a new crossing can be detected
        if not crossed:
            self.crossed_active.pop(track_id, None)

        event = None
        if crossed and previous is not None and track_id not in self.crossed_active:
            self.crossed_active[track_id] = True
            class_name = track.get("class_name")
            if class_name != "forklift_empty":
                direction = self._compute_direction(bottom, previous)
                event = create_event(
                    camera_id=self.camera_id,
                    track_id=track_id,
                    direction=direction,
                    bbox=track.get("bbox", []),
                    class_name=class_name,
                )

        self.last_debug_snapshot = self._debug_snapshot(track, center, bottom, crossed, event)
        return event

    def _compute_direction(
        self,
        point: tuple[float, float],
        previous: tuple[float, float],
    ) -> str:
        return "in" if dot_product(movement_vector(previous, point), self.in_direction) > 0 else "out"

    def prune_missing(self, frame_index: int | None = None) -> None:
        self._advance_frame(frame_index)
        stale_track_ids = [
            track_id
            for track_id, last_seen_frame in self.last_seen_frames.items()
            if self.current_frame - last_seen_frame > self.max_missing_frames
        ]
        for track_id in stale_track_ids:
            self.last_seen_frames.pop(track_id, None)
            self.last_centers.pop(track_id, None)
            self.crossed_active.pop(track_id, None)

    def _advance_frame(self, frame_index: int | None) -> None:
        if frame_index is None:
            self.current_frame += 1
            return
        self.current_frame = max(self.current_frame, int(frame_index))

    def _debug_snapshot(
        self,
        track: dict[str, Any],
        center: tuple[float, float],
        bottom: tuple[float, float],
        crossed: bool,
        event: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "frame": self.current_frame,
            "track_id": int(track["track_id"]),
            "bbox": track.get("bbox", []),
            "center": center,
            "bottom": bottom,
            "crossed": crossed,
            "class_name": track.get("class_name"),
            "event": event["direction"] if event else None,
        }

    @staticmethod
    def _center(value: Sequence[float]) -> tuple[float, float]:
        return float(value[0]), float(value[1])

    @staticmethod
    def _representative_points(track: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
        bbox = track.get("bbox")
        if bbox and len(bbox) >= 4:
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            return (cx, cy), (cx, float(bbox[3]))
        c = DirectionDetector._center(track["center"])
        return c, c
