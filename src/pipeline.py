from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from src.debug_video import DebugVideoWriter
from src.detector import ForkliftDetector
from src.detector_onnx import ONNXForkliftDetector
from src.direction import DirectionDetector
from src.tracker import ByteTrackTracker

DetectorFactory = Callable[..., Any]
TrackerFactory = Callable[[], Any]
CaptureFactory = Callable[[str], Any]
EventSink = Callable[[dict[str, Any]], None]
DebugSink = Callable[[str], None]
DebugVideoWriterFactory = Callable[..., Any]


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
    debug: bool = False,
    debug_every: int = 1,
    debug_sink: DebugSink | None = None,
    debug_video: bool = False,
    debug_video_writer_factory: DebugVideoWriterFactory | None = None,
    model_path: str | None = None,
    trt_fp16: bool = False,
    trt_max_workspace_size: int | None = None,
    trt_builder_optimization_level: int | None = None,
) -> None:
    config = load_config(config_path)
    detector_factory = detector_factory or _create_detector
    tracker_factory = tracker_factory or ByteTrackTracker
    capture_factory = capture_factory or _create_capture
    event_sink = event_sink or print
    debug_sink = debug_sink or print

    for camera_config in config["cameras"]:
        if model_path is not None:
            camera_config = {**camera_config, "model_path": model_path}
        _run_camera(
            camera_config,
            class_names=config["class_names"],
            detector_factory=detector_factory,
            tracker_factory=tracker_factory,
            capture_factory=capture_factory,
            event_sink=event_sink,
            debug=debug,
            debug_every=debug_every,
            debug_sink=debug_sink,
            debug_video=debug_video,
            debug_video_writer_factory=debug_video_writer_factory or DebugVideoWriter,
            trt_fp16=trt_fp16,
            trt_max_workspace_size=trt_max_workspace_size,
            trt_builder_optimization_level=trt_builder_optimization_level,
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = load_config_path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Config must define a non-empty cameras list")

    class_names = data.get("class_names", ["forklift_2"])
    if isinstance(class_names, str):
        class_names = [class_names]

    return {
        "class_names": list(class_names),
        "cameras": [_normalize_camera_config(camera) for camera in cameras],
    }


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
        "max_missing_frames": int(camera.get("max_missing_frames", 30)),
    }


def _run_camera(
    camera_config: dict[str, Any],
    *,
    class_names: list[str],
    detector_factory: DetectorFactory,
    tracker_factory: TrackerFactory,
    capture_factory: CaptureFactory,
    event_sink: EventSink,
    debug: bool = False,
    debug_every: int = 1,
    debug_sink: DebugSink | None = None,
    debug_video: bool = False,
    debug_video_writer_factory: DebugVideoWriterFactory | None = None,
    trt_fp16: bool = False,
    trt_max_workspace_size: int | None = None,
    trt_builder_optimization_level: int | None = None,
) -> None:
    detector_kwargs = {
        "model_path": camera_config["model_path"],
        "confidence": camera_config["confidence"],
        "class_names": class_names,
    }
    if trt_fp16:
        detector_kwargs["trt_fp16"] = True
    if trt_max_workspace_size is not None:
        detector_kwargs["trt_max_workspace_size"] = trt_max_workspace_size
    if trt_builder_optimization_level is not None:
        detector_kwargs["trt_builder_optimization_level"] = trt_builder_optimization_level
    detector = detector_factory(**detector_kwargs)
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

    debug_video_writer = None
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if debug_video and debug_video_writer is None and debug_video_writer_factory is not None:
                debug_video_writer = debug_video_writer_factory(
                    camera_config=camera_config,
                    capture=capture,
                    first_frame=frame,
                )

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            should_debug_frame = debug and _should_debug_frame(frame_index, debug_every, detections, tracks)
            if should_debug_frame and debug_sink is not None:
                debug_sink(
                    f"[debug] camera={camera_config['camera_id']} frame={frame_index} "
                    f"detections={len(detections)} tracks={len(tracks)}"
                )
            frame_events = []
            for track in tracks:
                event = direction_detector.update(track, frame_index=frame_index)
                if should_debug_frame and debug_sink is not None:
                    debug_sink(_format_track_debug(camera_config["camera_id"], direction_detector.last_debug_snapshot))
                if event is not None:
                    frame_events.append(event)
                    event_sink(event)
            if debug_video_writer is not None:
                debug_video_writer.write(
                    frame=frame,
                    frame_index=frame_index,
                    detections=detections,
                    tracks=tracks,
                    events=frame_events,
                )
            direction_detector.prune_missing(frame_index=frame_index)
    finally:
        if debug_video_writer is not None:
            debug_video_writer.release()
        capture.release()


def _create_detector(
    *,
    model_path: str,
    confidence: float,
    class_names: list[str],
    providers: list[str | tuple[str, dict[str, str]]] | None = None,
    trt_fp16: bool = False,
    trt_max_workspace_size: int | None = None,
    trt_builder_optimization_level: int | None = None,
    ort_profile: bool = False,
    ort_profile_prefix: str | None = None,
    ort_verbose: bool = False,
) -> Any:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        kwargs = {
            "onnx_path": model_path,
            "confidence": confidence,
            "allowed_class_names": class_names,
        }
        if providers is not None:
            kwargs["providers"] = providers
        if trt_fp16:
            kwargs["trt_fp16"] = True
        if trt_max_workspace_size is not None:
            kwargs["trt_max_workspace_size"] = trt_max_workspace_size
        if trt_builder_optimization_level is not None:
            kwargs["trt_builder_optimization_level"] = trt_builder_optimization_level
        if ort_profile:
            kwargs["ort_profile"] = True
        if ort_profile_prefix is not None:
            kwargs["ort_profile_prefix"] = ort_profile_prefix
        if ort_verbose:
            kwargs["ort_verbose"] = True
        return ONNXForkliftDetector(**kwargs)
    if suffix == ".engine":
        raise ValueError(
            "TensorRT .engine models are no longer supported; export or deploy an equivalent .onnx model instead."
        )
    return ForkliftDetector(model_path=model_path, confidence=confidence, class_names=class_names)


def _create_capture(source: str) -> Any:
    import cv2

    return cv2.VideoCapture(source)


def _should_debug_frame(frame_index: int, debug_every: int, detections: list[Any], tracks: list[Any]) -> bool:
    return frame_index % max(1, int(debug_every)) == 0 or bool(detections) or bool(tracks)


def _format_track_debug(camera_id: str, snapshot: dict[str, Any] | None) -> str:
    if snapshot is None:
        return f"[debug] camera={camera_id} track=None"
    return (
        f"[debug] camera={camera_id} frame={snapshot['frame']} track_id={snapshot['track_id']} "
        f"bbox={snapshot['bbox']} center={snapshot['center']} bottom={snapshot['bottom']} "
        f"crossed={snapshot['crossed']} class={snapshot.get('class_name')} event={snapshot['event']}"
    )


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config field: {key}")
    return mapping[key]


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numeric values")
    return float(value[0]), float(value[1])
