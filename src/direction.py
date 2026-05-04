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
    last_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    emitted_directions: dict[int, set[str]] = field(default_factory=dict)

    def update(self, track: dict[str, Any]) -> dict[str, Any] | None:
        track_id = int(track["track_id"])
        center = self._center(track["center"])
        previous = self.last_centers.get(track_id)
        self.last_centers[track_id] = center

        if previous is None:
            return None
        if not crossed_line(previous, center, self.line_start, self.line_end):
            return None

        direction = "in" if dot_product(movement_vector(previous, center), self.in_direction) > 0 else "out"
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

    @staticmethod
    def _center(value: Sequence[float]) -> tuple[float, float]:
        return float(value[0]), float(value[1])
