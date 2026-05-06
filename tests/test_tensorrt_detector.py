import unittest

import numpy as np

from src.detector_tensorrt import TensorRTForkliftDetector


class TensorRTForkliftDetectorTest(unittest.TestCase):
    def test_detect_returns_filtered_deploy_format_detections(self):
        detector = TensorRTForkliftDetector(
            "models/model.engine",
            confidence=0.5,
            class_names=["forklift_with_load", "forklift_empty"],
            allowed_class_names=["forklift_empty"],
            runtime_loader=lambda path: FakeTensorRTRuntime(
                input_shape=(1, 3, 512, 512),
                output_names=("scores", "labels", "boxes"),
                outputs={
                    "scores": np.array([[0.95, 0.40, 0.88]], dtype=np.float32),
                    "labels": np.array([[0, 0, 1]], dtype=np.int64),
                    "boxes": np.array(
                        [[[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 8.0, 9.0], [5.0, 6.0, 12.0, 18.0]]],
                        dtype=np.float32,
                    ),
                },
            ),
        )
        detector._preprocess = lambda frame: np.zeros((1, 3, 512, 512), dtype=np.float32)

        detections = detector.detect(np.zeros((32, 64, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [5.0, 6.0, 12.0, 18.0])
        self.assertEqual(detections[0]["class_name"], "forklift_empty")
        self.assertAlmostEqual(detections[0]["score"], 0.88, places=5)

    def test_init_raises_clear_error_for_unsupported_input_shape(self):
        with self.assertRaisesRegex(ValueError, "1x3x512x512"):
            TensorRTForkliftDetector(
                "models/model.engine",
                runtime_loader=lambda path: FakeTensorRTRuntime(
                    input_shape=(1, 3, 640, 640),
                    output_names=("scores", "labels", "boxes"),
                    outputs={},
                ),
            )

    def test_init_raises_clear_error_for_missing_deploy_outputs(self):
        with self.assertRaisesRegex(ValueError, "scores, labels, boxes"):
            TensorRTForkliftDetector(
                "models/model.engine",
                runtime_loader=lambda path: FakeTensorRTRuntime(
                    input_shape=(1, 3, 512, 512),
                    output_names=("dets", "labels"),
                    outputs={},
                ),
            )

    def test_init_raises_clear_error_when_runtime_is_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "TensorRT runtime is unavailable"):
            TensorRTForkliftDetector(
                "models/model.engine",
                runtime_loader=lambda path: (_ for _ in ()).throw(ImportError("No module named tensorrt")),
            )


class FakeTensorRTRuntime:
    def __init__(self, *, input_shape, output_names, outputs):
        self._input_shape = input_shape
        self._output_names = tuple(output_names)
        self._outputs = outputs
        self.last_inputs = None

    def get_input_shape(self):
        return self._input_shape

    def get_output_names(self):
        return self._output_names

    def infer(self, inputs):
        self.last_inputs = inputs
        return self._outputs


if __name__ == "__main__":
    unittest.main()
