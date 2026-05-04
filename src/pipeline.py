from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from src.detector import ForkliftDetector
from src.direction import DirectionDetector
from src.tracker import ByteTrackTracker

DetectorFactory = Callable[..., Any]
TrackerFactory = Callable[[], Any]
CaptureFactory = Callable[[str], Any]
EventSink = Callable[[dict[str, Any]], None]


def load_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def run(
    config_path: str | Path,
    *,
    detector_factory: DetectorFactory | None = None,
    tracker_factory: TrackerFactory | None = None,
    capture_factory: CaptureFactory | None = None,
    event_sink: EventSink | None = None,
) -> None:
    config = load_config(config_path)
    detector_factory = detector_factory or _create_detector
    tracker_factory = tracker_factory or ByteTrackTracker
    capture_factory = capture_factory or _create_capture
    event_sink = event_sink or print

    for camera_config in config["cameras"]:
        _run_camera(
            camera_config,
            detector_factory=detector_factory,
            tracker_factory=tracker_factory,
            capture_factory=capture_factory,
            event_sink=event_sink,
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = load_config_path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Config must define a non-empty cameras list")

    return {"cameras": [_normalize_camera_config(camera) for camera in cameras]}


def _normalize_camera_config(camera: Any) -> dict[str, Any]:
    if not isinstance(camera, dict):
        raise ValueError("Each camera config must be a mapping")

    camera_id = _required(camera, "camera_id")
    source = _required(camera, "source")
    line = _required(camera, "line")
    if not isinstance(line, dict):
        raise ValueError(f"Camera {camera_id} line must be a mapping")

    return {
        "camera_id": str(camera_id),
        "source": str(source),
        "line_start": _point(_required(line, "start"), f"Camera {camera_id} line.start"),
        "line_end": _point(_required(line, "end"), f"Camera {camera_id} line.end"),
        "line_width": float(camera.get("line_width", 40)),
        "in_direction": _point(_required(camera, "in_direction"), f"Camera {camera_id} in_direction"),
        "model_path": str(camera.get("model_path", "models/best.pt")),
        "confidence": float(camera.get("confidence", 0.4)),
        "class_name": str(camera.get("class_name", "forklift_2")),
        "max_missing_frames": int(camera.get("max_missing_frames", 30)),
    }


def _run_camera(
    camera_config: dict[str, Any],
    *,
    detector_factory: DetectorFactory,
    tracker_factory: TrackerFactory,
    capture_factory: CaptureFactory,
    event_sink: EventSink,
) -> None:
    detector = detector_factory(
        model_path=camera_config["model_path"],
        confidence=camera_config["confidence"],
        class_name=camera_config["class_name"],
    )
    tracker = tracker_factory()
    direction_detector = DirectionDetector(
        camera_id=camera_config["camera_id"],
        line_start=camera_config["line_start"],
        line_end=camera_config["line_end"],
        in_direction=camera_config["in_direction"],
        max_missing_frames=camera_config["max_missing_frames"],
        line_width=camera_config["line_width"],
    )
    capture = capture_factory(camera_config["source"])

    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video source for camera {camera_config['camera_id']}: {camera_config['source']}"
        )

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            for track in tracks:
                event = direction_detector.update(track, frame_index=frame_index)
                if event is not None:
                    event_sink(event)
            direction_detector.prune_missing(frame_index=frame_index)
    finally:
        capture.release()


def _create_detector(*, model_path: str, confidence: float, class_name: str) -> ForkliftDetector:
    return ForkliftDetector(model_path=model_path, confidence=confidence, class_name=class_name)


def _create_capture(source: str) -> Any:
    import cv2

    return cv2.VideoCapture(source)


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config field: {key}")
    return mapping[key]


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numeric values")
    return float(value[0]), float(value[1])
