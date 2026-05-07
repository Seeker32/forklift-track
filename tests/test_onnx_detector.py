import unittest
from pathlib import Path

import numpy as np

from src.detector_onnx import ONNXForkliftDetector


class ONNXForkliftDetectorTest(unittest.TestCase):
    def test_init_prefers_tensorrt_then_cuda_then_cpu_providers(self):
        fake_session = FakeSession(output_names=("scores", "labels", "boxes"))
        fake_runtime = FakeRuntimeModule(fake_session)
        expected_cache_path = str(Path(__file__).resolve().parents[1] / "models" / "trt_engine_cache")

        ONNXForkliftDetector("models/model.onnx", runtime_module=fake_runtime)

        self.assertEqual(fake_runtime.calls[0]["path"], "models/model.onnx")
        self.assertEqual(
            fake_runtime.calls[0]["providers"],
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": expected_cache_path,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_init_passes_tensorrt_provider_options_when_configured(self):
        fake_session = FakeSession(output_names=("scores", "labels", "boxes"))
        fake_runtime = FakeRuntimeModule(fake_session)

        ONNXForkliftDetector(
            "models/model.onnx",
            trt_engine_cache_path="cache/trt",
            trt_fp16=True,
            trt_max_workspace_size=2147483648,
            trt_builder_optimization_level=5,
            runtime_module=fake_runtime,
        )

        self.assertEqual(
            fake_runtime.calls[0]["providers"],
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": "cache/trt",
                        "trt_fp16_enable": "True",
                        "trt_max_workspace_size": "2147483648",
                        "trt_builder_optimization_level": "5",
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_init_merges_tensorrt_options_when_explicit_trt_provider_is_requested(self):
        fake_session = FakeSession(output_names=("scores", "labels", "boxes"))
        fake_runtime = FakeRuntimeModule(fake_session)
        expected_cache_path = str(Path(__file__).resolve().parents[1] / "models" / "trt_engine_cache")

        ONNXForkliftDetector(
            "models/model.onnx",
            providers=["TensorrtExecutionProvider"],
            trt_fp16=True,
            trt_max_workspace_size=2147483648,
            trt_builder_optimization_level=5,
            runtime_module=fake_runtime,
        )

        self.assertEqual(
            fake_runtime.calls[0]["providers"],
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "True",
                        "trt_engine_cache_path": expected_cache_path,
                        "trt_fp16_enable": "True",
                        "trt_max_workspace_size": "2147483648",
                        "trt_builder_optimization_level": "5",
                    },
                )
            ],
        )

    def test_init_configures_ort_diagnostics_when_enabled(self):
        fake_session = FakeSession(output_names=("scores", "labels", "boxes"))
        fake_runtime = FakeRuntimeModule(fake_session)

        ONNXForkliftDetector(
            "models/model.onnx",
            runtime_module=fake_runtime,
            ort_profile=True,
            ort_profile_prefix="outputs/profile",
            ort_verbose=True,
        )

        session_options = fake_runtime.calls[0]["sess_options"]
        self.assertTrue(session_options.enable_profiling)
        self.assertEqual(session_options.profile_file_prefix, "outputs/profile")
        self.assertEqual(session_options.log_severity_level, 0)
        self.assertGreaterEqual(session_options.log_verbosity_level, 1)

    def test_detect_returns_filtered_raw_output_detections(self):
        fake_session = FakeSession(
            output_names=("pred_boxes", "pred_logits"),
            outputs=[
                np.array(
                    [[[0.20, 0.25, 0.10, 0.20], [0.50, 0.50, 0.20, 0.20], [0.70, 0.50, 0.20, 0.30]]],
                    dtype=np.float32,
                ),
                np.array(
                    [[[-5.0, -2.0, -2.0], [-4.0, -3.0, -0.2], [-3.0, 2.5, -3.0]]],
                    dtype=np.float32,
                ),
            ],
        )
        fake_runtime = FakeRuntimeModule(fake_session)

        detector = ONNXForkliftDetector(
            "models/model.onnx",
            confidence=0.5,
            class_names=["forklift_with_load", "forklift_empty"],
            allowed_class_names=["forklift_empty"],
            runtime_module=fake_runtime,
        )

        detector._preprocess = lambda frame: np.zeros((1, 3, 512, 512), dtype=np.float32)
        detections = detector.detect(np.zeros((32, 64, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        for actual, expected in zip(detections[0]["bbox"], [38.4, 11.2, 51.2, 20.8]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(detections[0]["class_name"], "forklift_empty")
        self.assertAlmostEqual(detections[0]["score"], 0.9241418, places=5)
        self.assertEqual(len(fake_session.calls), 1)
        self.assertIsNone(fake_session.calls[0]["output_names"])
        np.testing.assert_array_equal(
            fake_session.calls[0]["inputs"]["input"],
            np.zeros((1, 3, 512, 512), dtype=np.float32),
        )

    def test_init_raises_clear_error_when_onnxruntime_is_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "ONNX Runtime is unavailable"):
            ONNXForkliftDetector("models/model.onnx", runtime_loader=lambda: (_ for _ in ()).throw(ModuleNotFoundError()))

    def test_detect_reports_stage_timings_when_sink_is_configured(self):
        fake_session = FakeSession(
            output_names=("scores", "labels", "boxes"),
            outputs=[
                np.array([[0.9]], dtype=np.float32),
                np.array([[0]], dtype=np.int64),
                np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32),
            ],
        )
        fake_runtime = FakeRuntimeModule(fake_session)
        recorded_timings = []

        detector = ONNXForkliftDetector(
            "models/model.onnx",
            runtime_module=fake_runtime,
            clock=FakeClock([1.0, 1.004, 1.014, 1.020]),
        )

        detector._preprocess = lambda frame: np.zeros((1, 3, 512, 512), dtype=np.float32)
        detector.set_stage_timing_sink(recorded_timings.append)
        detections = detector.detect(np.zeros((32, 64, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(len(recorded_timings), 1)
        self.assertAlmostEqual(recorded_timings[0]["preprocess"], 4.0, places=5)
        self.assertAlmostEqual(recorded_timings[0]["inference"], 10.0, places=5)
        self.assertAlmostEqual(recorded_timings[0]["postprocess"], 6.0, places=5)


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, *, output_names, outputs=None):
        self._outputs = [FakeOutput(name) for name in output_names]
        self._run_outputs = outputs or []
        self.calls = []

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, inputs):
        self.calls.append({"output_names": output_names, "inputs": inputs})
        return self._run_outputs


class FakeRuntimeModule:
    def __init__(self, session):
        self._session = session
        self.calls = []
        self.SessionOptions = FakeSessionOptions

    def InferenceSession(self, path, *, providers, sess_options=None):
        self.calls.append({"path": path, "providers": providers, "sess_options": sess_options})
        return self._session


class FakeClock:
    def __init__(self, values):
        self._values = list(values)

    def __call__(self):
        return self._values.pop(0)


class FakeSessionOptions:
    def __init__(self):
        self.enable_profiling = False
        self.profile_file_prefix = "onnxruntime_profile_"
        self.log_severity_level = -1
        self.log_verbosity_level = 0


if __name__ == "__main__":
    unittest.main()
