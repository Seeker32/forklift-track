import unittest

from src.detector_tensorrt import TensorRTForkliftDetector


class TensorRTForkliftDetectorTest(unittest.TestCase):
    def test_init_raises_clear_error_for_removed_engine_support(self):
        with self.assertRaisesRegex(RuntimeError, r"\.engine.*\.onnx"):
            TensorRTForkliftDetector("models/model.engine")


if __name__ == "__main__":
    unittest.main()
