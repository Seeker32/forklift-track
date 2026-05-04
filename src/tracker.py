from __future__ import annotations

from typing import Any


class ByteTrackTracker:
    """Placeholder tracker interface for the ByteTrack integration in phase 4."""

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError("ByteTrack integration is implemented in phase 4.")
