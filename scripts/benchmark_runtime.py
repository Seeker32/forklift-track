"""Benchmark detector-only and full-pipeline runtime on a real video source.

Usage:
  PYTHONPATH=. uv run python scripts/benchmark_runtime.py --mode detector
  PYTHONPATH=. uv run python scripts/benchmark_runtime.py --mode pipeline --camera-id gate_01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline import _create_capture, _create_detector, _run_camera, load_config
from src.tracker import ByteTrackTracker


def run_detector_benchmark(
    *,
    source: str,
    model_path: str,
    confidence: float,
    class_names: list[str],
    frames: int,
    warmup: int,
    detector_factory: Any = _create_detector,
    capture_factory: Any = _create_capture,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    detector = detector_factory(model_path=model_path, confidence=confidence, class_names=class_names)
    capture = capture_factory(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    providers = _detector_providers(detector)
    processed_frames = 0
    total_detections = 0
    class_counts = {name: 0 for name in class_names}
    latencies_ms: list[float] = []

    loop_start = clock()
    last_timestamp = loop_start
    max_frames = _max_frames(frames=frames, warmup=warmup)

    try:
        while True:
            if max_frames and processed_frames >= max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            detect_start = clock()
            detections = detector.detect(frame)
            detect_end = clock()
            last_timestamp = detect_end

            processed_frames += 1
            total_detections += len(detections)
            _accumulate_class_counts(class_counts, detections)

            if processed_frames > warmup:
                latencies_ms.append((detect_end - detect_start) * 1000.0)
    finally:
        capture.release()

    return {
        "mode": "detector",
        "source": source,
        "model_path": model_path,
        "providers": providers,
        "processed_frames": processed_frames,
        "warmup_frames": min(warmup, processed_frames),
        "measured_frames": max(0, processed_frames - warmup),
        "total_detections": total_detections,
        "class_counts": class_counts,
        "total_time_s": max(0.0, last_timestamp - loop_start),
        "avg_fps": _avg_fps(processed_frames, last_timestamp - loop_start),
        "latency_ms": _latency_summary(latencies_ms),
    }


def run_pipeline_benchmark(
    *,
    camera_config: dict[str, Any],
    class_names: list[str],
    frames: int,
    warmup: int,
    detector_factory: Any = _create_detector,
    tracker_factory: Any = ByteTrackTracker,
    capture_factory: Any = _create_capture,
    run_camera: Any = _run_camera,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    metrics = _PipelineMetrics(clock=clock, class_names=class_names, warmup=warmup)
    max_frames = _max_frames(frames=frames, warmup=warmup)

    def wrapped_detector_factory(*, model_path: str, confidence: float, class_names: list[str]) -> Any:
        detector = detector_factory(model_path=model_path, confidence=confidence, class_names=class_names)
        metrics.providers = _detector_providers(detector)
        return _TimedDetector(detector=detector, metrics=metrics)

    def wrapped_capture_factory(source: str) -> Any:
        capture = capture_factory(source)
        return _LimitedCapture(capture=capture, max_frames=max_frames)

    run_camera(
        camera_config,
        class_names=class_names,
        detector_factory=wrapped_detector_factory,
        tracker_factory=tracker_factory,
        capture_factory=wrapped_capture_factory,
        event_sink=metrics.record_event,
    )

    return {
        "mode": "pipeline",
        "camera_id": camera_config["camera_id"],
        "source": camera_config["source"],
        "model_path": camera_config["model_path"],
        "providers": metrics.providers,
        "processed_frames": metrics.processed_frames,
        "warmup_frames": min(warmup, metrics.processed_frames),
        "measured_frames": max(0, metrics.processed_frames - warmup),
        "event_count": metrics.event_count,
        "total_detections": metrics.total_detections,
        "class_counts": metrics.class_counts,
        "total_time_s": max(0.0, metrics.total_time_s),
        "avg_fps": _avg_fps(metrics.processed_frames, metrics.total_time_s),
        "latency_ms": _latency_summary(metrics.latencies_ms),
    }


class _PipelineMetrics:
    def __init__(self, *, clock: Any, class_names: list[str], warmup: int) -> None:
        self.clock = clock
        self.class_counts = {name: 0 for name in class_names}
        self.warmup = warmup
        self.providers: list[str] = []
        self.processed_frames = 0
        self.total_detections = 0
        self.event_count = 0
        self.latencies_ms: list[float] = []
        self._loop_start: float | None = None
        self._last_timestamp: float | None = None

    @property
    def total_time_s(self) -> float:
        if self._loop_start is None or self._last_timestamp is None:
            return 0.0
        return self._last_timestamp - self._loop_start

    def record_detect(self, detections: list[dict[str, Any]], *, latency_ms: float, timestamp: float) -> None:
        if self._loop_start is None:
            self._loop_start = timestamp - (latency_ms / 1000.0)
        self._last_timestamp = timestamp
        self.processed_frames += 1
        self.total_detections += len(detections)
        _accumulate_class_counts(self.class_counts, detections)
        if self.processed_frames > self.warmup:
            self.latencies_ms.append(latency_ms)

    def record_event(self, event: dict[str, Any]) -> None:
        self.event_count += 1


class _TimedDetector:
    def __init__(self, *, detector: Any, metrics: _PipelineMetrics) -> None:
        self._detector = detector
        self._metrics = metrics

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        start = self._metrics.clock()
        detections = self._detector.detect(frame)
        end = self._metrics.clock()
        self._metrics.record_detect(detections, latency_ms=(end - start) * 1000.0, timestamp=end)
        return detections


class _LimitedCapture:
    def __init__(self, *, capture: Any, max_frames: int) -> None:
        self._capture = capture
        self._max_frames = max_frames
        self._frames_read = 0

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def read(self) -> tuple[bool, Any]:
        if self._max_frames and self._frames_read >= self._max_frames:
            return False, None
        ok, frame = self._capture.read()
        if ok:
            self._frames_read += 1
        return ok, frame

    def release(self) -> None:
        self._capture.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capture, name)


def _max_frames(*, frames: int, warmup: int) -> int:
    if frames <= 0:
        return 0
    return frames + max(0, warmup)


def _accumulate_class_counts(class_counts: dict[str, int], detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        class_name = detection.get("class_name")
        if class_name is None:
            continue
        class_counts[class_name] = class_counts.get(class_name, 0) + 1


def _detector_providers(detector: Any) -> list[str]:
    session = getattr(detector, "_session", None)
    get_providers = getattr(session, "get_providers", None)
    if callable(get_providers):
        return list(get_providers())
    return []


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _avg_fps(frame_count: int, total_time_s: float) -> float:
    if frame_count <= 0 or total_time_s <= 0:
        return 0.0
    return float(frame_count / total_time_s)


def _resolve_camera(config: dict[str, Any], *, camera_id: str | None, source: str | None) -> dict[str, Any]:
    cameras = list(config["cameras"])
    if camera_id is not None:
        for camera in cameras:
            if camera["camera_id"] == camera_id:
                selected = dict(camera)
                break
        else:
            raise ValueError(f"Camera not found: {camera_id}")
    else:
        selected = dict(cameras[0])

    if source is not None:
        selected["source"] = source
    return selected


def _print_summary(result: dict[str, Any]) -> None:
    print(f"mode:            {result['mode']}")
    if "camera_id" in result:
        print(f"camera_id:       {result['camera_id']}")
    print(f"source:          {result['source']}")
    print(f"model_path:      {result['model_path']}")
    if result.get("providers"):
        print(f"providers:       {', '.join(result['providers'])}")
    print(f"processed_frames:{result['processed_frames']}")
    print(f"warmup_frames:   {result['warmup_frames']}")
    print(f"measured_frames: {result['measured_frames']}")
    print(f"total_time_s:    {result['total_time_s']:.3f}")
    print(f"avg_fps:         {result['avg_fps']:.2f}")
    print(f"total_detections:{result['total_detections']}")
    if "event_count" in result:
        print(f"event_count:     {result['event_count']}")
    print("latency_ms:")
    print(f"  mean: {result['latency_ms']['mean']:.2f}")
    print(f"  p50:  {result['latency_ms']['p50']:.2f}")
    print(f"  p95:  {result['latency_ms']['p95']:.2f}")
    print(f"  p99:  {result['latency_ms']['p99']:.2f}")
    print("class_counts:")
    for class_name, count in result["class_counts"].items():
        print(f"  {class_name}: {count}")


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark real detector and pipeline runtime")
    parser.add_argument("--mode", choices=["detector", "pipeline"], default="detector")
    parser.add_argument("--config", default=str(project_root / "config" / "cameras.yaml"))
    parser.add_argument("--camera-id", help="Benchmark one configured camera by camera_id")
    parser.add_argument("--source", help="Override video source for the selected camera")
    parser.add_argument("--model-path", help="Override model path for the selected camera")
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warmup (0=all)")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames to exclude from latency stats")
    parser.add_argument("--confidence", type=float, help="Override detector confidence")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    camera_config = _resolve_camera(config, camera_id=args.camera_id, source=args.source)
    if args.model_path is not None:
        camera_config["model_path"] = args.model_path
    if args.confidence is not None:
        camera_config["confidence"] = args.confidence

    if args.mode == "detector":
        result = run_detector_benchmark(
            source=camera_config["source"],
            model_path=camera_config["model_path"],
            confidence=camera_config["confidence"],
            class_names=config["class_names"],
            frames=args.frames,
            warmup=args.warmup,
        )
    else:
        result = run_pipeline_benchmark(
            camera_config=camera_config,
            class_names=config["class_names"],
            frames=args.frames,
            warmup=args.warmup,
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return
    _print_summary(result)


if __name__ == "__main__":
    main()
