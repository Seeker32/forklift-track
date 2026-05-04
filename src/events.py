from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def create_event(
    *,
    camera_id: str,
    track_id: int,
    direction: str,
    bbox: list[float] | tuple[float, ...],
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    event_time = timestamp or datetime.now(timezone.utc)
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "direction": direction,
        "timestamp": event_time.isoformat(),
        "bbox": list(bbox),
    }
