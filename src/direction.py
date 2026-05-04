from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.events import create_event
from src.geometry import dot_product, line_zone_side, movement_vector


@dataclass
class DirectionDetector:
    camera_id: str
    line_start: Sequence[float]
    line_end: Sequence[float]
    in_direction: Sequence[float]
    max_missing_frames: int = 30
    line_width: float = 40.0
    last_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    track_zone_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    emitted_directions: dict[int, set[str]] = field(default_factory=dict)
    last_seen_frames: dict[int, int] = field(default_factory=dict)
    current_frame: int = 0

    def update(self, track: dict[str, Any], frame_index: int | None = None) -> dict[str, Any] | None:
        self._advance_frame(frame_index)
        track_id = int(track["track_id"])
        point = self._representative_point(track)
        self.last_seen_frames[track_id] = self.current_frame
        self.prune_missing(self.current_frame)

        previous = self.last_centers.get(track_id)
        self.last_centers[track_id] = point

        side = line_zone_side(point, self.line_start, self.line_end, self.line_width)
        state = self.track_zone_states.get(track_id)
        if state is None:
            self.track_zone_states[track_id] = {
                "last_side": side,
                "entry_side": None,
                "entry_point": None,
                "inside_zone": side == 0,
            }
            return None

        event = None
        was_inside = bool(state["inside_zone"])
        last_side = int(state["last_side"])

        if side == 0:
            if not was_inside and last_side != 0:
                state["entry_side"] = last_side
                state["entry_point"] = point
            state["inside_zone"] = True
        else:
            if was_inside:
                entry_side = state["entry_side"]
                if entry_side is not None and side != entry_side:
                    event = self._create_crossing_event(track, point, previous, state)
                state["entry_side"] = None
                state["entry_point"] = None
            state["inside_zone"] = False

        state["last_side"] = side
        return event

    def _create_crossing_event(
        self,
        track: dict[str, Any],
        point: tuple[float, float],
        previous: tuple[float, float] | None,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        track_id = int(track["track_id"])
        entry_point = state.get("entry_point")
        start_point = entry_point if entry_point is not None else previous
        if start_point is None:
            return None

        direction = "in" if dot_product(movement_vector(start_point, point), self.in_direction) > 0 else "out"
        emitted = self.emitted_directions.setdefault(track_id, set())
        if direction in emitted:
            return None

        emitted.add(direction)
        return create_event(
            camera_id=self.camera_id,
            track_id=track_id,
            direction=direction,
            bbox=track.get("bbox", []),
        )

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
            self.track_zone_states.pop(track_id, None)
            self.emitted_directions.pop(track_id, None)

    def _advance_frame(self, frame_index: int | None) -> None:
        if frame_index is None:
            self.current_frame += 1
            return
        self.current_frame = max(self.current_frame, int(frame_index))

    @staticmethod
    def _center(value: Sequence[float]) -> tuple[float, float]:
        return float(value[0]), float(value[1])

    @staticmethod
    def _representative_point(track: dict[str, Any]) -> tuple[float, float]:
        bbox = track.get("bbox")
        if bbox and len(bbox) >= 4:
            return (float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3])
        return DirectionDetector._center(track["center"])
