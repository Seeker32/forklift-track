import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_runtime.py"
    spec = importlib.util.spec_from_file_location("benchmark_runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("onnxruntime", types.SimpleNamespace(preload_dlls=lambda: None))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkRuntimeScriptTest(unittest.TestCase):
    def test_run_detector_benchmark_reports_latency_and_frame_counts(self):
        benchmark = load_benchmark_module()
        clock = FakeClock([0.0, 1.0, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.112, 1.13, 1.16])
        detector = FakeDetector(
            stage_timings_ms=[
                {"preprocess": 4.0, "inference": 10.0, "postprocess": 6.0},
                {"preprocess": 5.0, "inference": 11.0, "postprocess": 8.0},
                {"preprocess": 7.0, "inference": 13.0, "postprocess": 9.0},
            ]
        )

        result = benchmark.run_detector_benchmark(
            source="video.mp4",
            model_path="models/inference_model.onnx",
            confidence=0.3,
            class_names=["forklift_with_load", "forklift_empty"],
            frames=2,
            warmup=1,
            detector_factory=lambda **kwargs: detector,
            capture_factory=lambda source: FakeCapture(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.zeros((4, 4, 3), dtype=np.uint8),
                ]
            ),
            clock=clock,
        )

        self.assertEqual(result["mode"], "detector")
        self.assertEqual(result["processed_frames"], 3)
        self.assertEqual(result["warmup_frames"], 1)
        self.assertEqual(result["measured_frames"], 2)
        self.assertEqual(result["total_detections"], 4)
        self.assertAlmostEqual(result["latency_ms"]["mean"], 25.0, places=5)
        self.assertAlmostEqual(result["latency_ms"]["p50"], 25.0, places=5)
        self.assertAlmostEqual(result["latency_ms"]["p95"], 29.5, places=5)
        self.assertEqual(result["class_counts"]["forklift_empty"], 2)
        self.assertAlmostEqual(result["stage_latency_ms"]["read"]["mean"], 11.0, places=5)
        self.assertAlmostEqual(result["stage_latency_ms"]["preprocess"]["mean"], 6.0, places=5)
        self.assertAlmostEqual(result["stage_latency_ms"]["inference"]["mean"], 12.0, places=5)
        self.assertAlmostEqual(result["stage_latency_ms"]["postprocess"]["mean"], 8.5, places=5)

    def test_run_pipeline_benchmark_wraps_real_flow_metrics(self):
        benchmark = load_benchmark_module()
        clock = FakeClock([10.0, 10.5, 10.52, 10.55, 10.59, 10.64, 10.70])

        result = benchmark.run_pipeline_benchmark(
            camera_config={
                "camera_id": "gate_01",
                "source": "video.mp4",
                "model_path": "models/inference_model.onnx",
                "confidence": 0.3,
                "line_start": (0.0, 0.0),
                "line_end": (1.0, 0.0),
                "line_width": 40.0,
                "in_direction": (0.0, 1.0),
                "max_missing_frames": 30,
            },
            class_names=["forklift_with_load", "forklift_empty"],
            frames=2,
            warmup=1,
            detector_factory=lambda **kwargs: FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: FakeCapture(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.zeros((4, 4, 3), dtype=np.uint8),
                ]
            ),
            run_camera=run_fake_camera,
            clock=clock,
        )

        self.assertEqual(result["mode"], "pipeline")
        self.assertEqual(result["processed_frames"], 3)
        self.assertEqual(result["warmup_frames"], 1)
        self.assertEqual(result["measured_frames"], 2)
        self.assertEqual(result["event_count"], 1)
        self.assertAlmostEqual(result["latency_ms"]["mean"], 40.0, places=5)
        self.assertEqual(result["total_detections"], 4)
        self.assertEqual(result["class_counts"]["forklift_with_load"], 2)

    def test_run_detector_benchmark_passes_selected_provider_mode_to_detector_factory(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        benchmark.run_detector_benchmark(
            source="video.mp4",
            model_path="models/inference_model.onnx",
            confidence=0.3,
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            provider_mode="cuda",
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
        )

        self.assertEqual(detector_kwargs[0]["providers"], ["CUDAExecutionProvider"])

    def test_run_detector_benchmark_passes_trt_fp16_to_detector_factory(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        benchmark.run_detector_benchmark(
            source="video.mp4",
            model_path="models/inference_model.onnx",
            confidence=0.3,
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            trt_fp16=True,
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
        )

        self.assertTrue(detector_kwargs[0]["trt_fp16"])

    def test_run_detector_benchmark_passes_trt_builder_options_to_detector_factory(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        benchmark.run_detector_benchmark(
            source="video.mp4",
            model_path="models/inference_model.onnx",
            confidence=0.3,
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            trt_max_workspace_size=2147483648,
            trt_builder_optimization_level=5,
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
        )

        self.assertEqual(detector_kwargs[0]["trt_max_workspace_size"], 2147483648)
        self.assertEqual(detector_kwargs[0]["trt_builder_optimization_level"], 5)

    def test_run_detector_benchmark_passes_ort_diagnostics_to_detector_factory_and_reports_profile(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        result = benchmark.run_detector_benchmark(
            source="video.mp4",
            model_path="models/inference_model.onnx",
            confidence=0.3,
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            ort_profile=True,
            ort_verbose=True,
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(
                profile_file="outputs/ort_profile.json"
            ),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
        )

        self.assertTrue(detector_kwargs[0]["ort_profile"])
        self.assertTrue(detector_kwargs[0]["ort_verbose"])
        self.assertEqual(result["ort_profile_file"], "outputs/ort_profile.json")

    def test_run_pipeline_benchmark_passes_selected_provider_mode_to_detector_factory(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        benchmark.run_pipeline_benchmark(
            camera_config={
                "camera_id": "gate_01",
                "source": "video.mp4",
                "model_path": "models/inference_model.onnx",
                "confidence": 0.3,
                "line_start": (0.0, 0.0),
                "line_end": (1.0, 0.0),
                "line_width": 40.0,
                "in_direction": (0.0, 1.0),
                "max_missing_frames": 30,
            },
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            provider_mode="cpu",
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
            run_camera=run_fake_camera,
        )

        self.assertEqual(detector_kwargs[0]["providers"], ["CPUExecutionProvider"])

    def test_run_pipeline_benchmark_passes_trt_fp16_to_detector_factory(self):
        benchmark = load_benchmark_module()
        detector_kwargs = []

        benchmark.run_pipeline_benchmark(
            camera_config={
                "camera_id": "gate_01",
                "source": "video.mp4",
                "model_path": "models/inference_model.onnx",
                "confidence": 0.3,
                "line_start": (0.0, 0.0),
                "line_end": (1.0, 0.0),
                "line_width": 40.0,
                "in_direction": (0.0, 1.0),
                "max_missing_frames": 30,
            },
            class_names=["forklift_with_load", "forklift_empty"],
            frames=1,
            warmup=0,
            trt_fp16=True,
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8)]),
            run_camera=run_fake_camera,
        )

        self.assertTrue(detector_kwargs[0]["trt_fp16"])


class FakeClock:
    def __init__(self, values):
        self._values = list(values)

    def __call__(self):
        return self._values.pop(0)


class FakeDetector:
    def __init__(self, stage_timings_ms=None, profile_file=None):
        self.calls = 0
        self.stage_timings_ms = list(stage_timings_ms or [])
        self._stage_timing_sink = None
        self.profile_file = profile_file

    def set_stage_timing_sink(self, sink):
        self._stage_timing_sink = sink

    def detect(self, frame):
        self.calls += 1
        if self._stage_timing_sink is not None and self.stage_timings_ms:
            self._stage_timing_sink(dict(self.stage_timings_ms[self.calls - 1]))
        if self.calls == 1:
            return [{"bbox": [0, 0, 1, 1], "score": 0.9, "class_name": "forklift_with_load"}]
        if self.calls == 2:
            return [{"bbox": [0, 0, 1, 1], "score": 0.8, "class_name": "forklift_empty"}]
        return [
            {"bbox": [0, 0, 1, 1], "score": 0.7, "class_name": "forklift_with_load"},
            {"bbox": [1, 1, 2, 2], "score": 0.6, "class_name": "forklift_empty"},
        ]

    def end_profiling(self):
        return self.profile_file


class FakeTracker:
    def update(self, detections):
        return [
            {
                "track_id": 1,
                "bbox": detections[0]["bbox"],
                "center": [0.5, 0.5],
                "score": detections[0]["score"],
                "class_name": detections[0]["class_name"],
            }
        ]


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def run_fake_camera(
    camera_config,
    *,
    class_names,
    detector_factory,
    tracker_factory,
    capture_factory,
    event_sink,
    debug=False,
    debug_every=1,
    debug_sink=None,
    debug_video=False,
    debug_video_writer_factory=None,
):
    detector = detector_factory(
        model_path=camera_config["model_path"],
        confidence=camera_config["confidence"],
        class_names=class_names,
    )
    tracker = tracker_factory()
    capture = capture_factory(camera_config["source"])

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            if tracks and detections[0]["class_name"] == "forklift_empty":
                event_sink(
                    {
                        "camera_id": camera_config["camera_id"],
                        "track_id": tracks[0]["track_id"],
                        "direction": "in",
                        "bbox": tracks[0]["bbox"],
                        "class_name": tracks[0]["class_name"],
                    }
                )
    finally:
        capture.release()


if __name__ == "__main__":
    unittest.main()
