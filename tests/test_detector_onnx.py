import unittest

import numpy as np

from src.detector_onnx import ONNXForkliftDetector


class OnnxForkliftDetectorTest(unittest.TestCase):
    def test_postprocess_non_deploy_uses_application_classes_without_background_offset(self):
        detector = ONNXForkliftDetector.__new__(ONNXForkliftDetector)
        detector.confidence = 0.3
        detector.class_names = ["forklift_with_load", "forklift_empty"]
        detector.num_select = 300

        dets = np.array(
            [
                [
                    [0.5, 0.5, 0.2, 0.4],
                    [0.25, 0.25, 0.2, 0.2],
                ]
            ],
            dtype=np.float32,
        )
        logits = np.array(
            [
                [
                    [5.0, -5.0, 10.0],
                    [-5.0, 4.0, 9.0],
                ]
            ],
            dtype=np.float32,
        )

        detections = detector._postprocess(dets, logits, orig_h=100, orig_w=200)

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "forklift_with_load")
        self.assertEqual(detections[1]["class_name"], "forklift_empty")
        np.testing.assert_allclose(detections[0]["bbox"], [80.0, 30.0, 120.0, 70.0])
        np.testing.assert_allclose(detections[1]["bbox"], [30.0, 15.0, 70.0, 35.0])

    def test_postprocess_non_deploy_raises_when_model_exposes_too_few_class_logits(self):
        detector = ONNXForkliftDetector.__new__(ONNXForkliftDetector)
        detector.confidence = 0.3
        detector.class_names = ["forklift_with_load", "forklift_empty"]
        detector.num_select = 300

        dets = np.zeros((1, 1, 4), dtype=np.float32)
        logits = np.zeros((1, 1, 1), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "expected at least 2"):
            detector._postprocess(dets, logits, orig_h=100, orig_w=200)


if __name__ == "__main__":
    unittest.main()
