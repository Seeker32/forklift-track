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
                        "trt_engine_cache_enable": "1",
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
            runtime_module=fake_runtime,
        )

        self.assertEqual(
            fake_runtime.calls[0]["providers"],
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "1",
                        "trt_engine_cache_path": "cache/trt",
                        "trt_fp16_enable": "1",
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_detect_returns_filtered_deploy_format_detections(self):
        fake_session = FakeSession(
            output_names=("scores", "labels", "boxes"),
            outputs=[
                np.array([[0.95, 0.40, 0.88]], dtype=np.float32),
                np.array([[0, 0, 1]], dtype=np.int64),
                np.array(
                    [[[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 8.0, 9.0], [5.0, 6.0, 12.0, 18.0]]],
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
        self.assertEqual(detections[0]["bbox"], [5.0, 6.0, 12.0, 18.0])
        self.assertEqual(detections[0]["class_name"], "forklift_empty")
        self.assertAlmostEqual(detections[0]["score"], 0.88, places=5)
        self.assertEqual(len(fake_session.calls), 1)
        self.assertEqual(fake_session.calls[0]["output_names"], ["scores", "labels", "boxes"])
        np.testing.assert_array_equal(
            fake_session.calls[0]["inputs"]["target_sizes"],
            np.array([[32, 64]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            fake_session.calls[0]["inputs"]["input"],
            np.zeros((1, 3, 512, 512), dtype=np.float32),
        )

    def test_init_raises_clear_error_when_onnxruntime_is_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "ONNX Runtime is unavailable"):
            ONNXForkliftDetector("models/model.onnx", runtime_loader=lambda: (_ for _ in ()).throw(ModuleNotFoundError()))


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

    def InferenceSession(self, path, *, providers):
        self.calls.append({"path": path, "providers": providers})
        return self._session


if __name__ == "__main__":
    unittest.main()
