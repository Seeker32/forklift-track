import unittest
import warnings

from src.detector import ForkliftDetector


class FakeBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [xyxy]
        self.conf = [confidence]
        self.cls = [class_id]


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeModel:
    names = {0: "person", 1: "forklift"}

    def __init__(self):
        self.calls = []

    def __call__(self, frame, conf, verbose):
        self.calls.append({"frame": frame, "conf": conf, "verbose": verbose})
        return [
            FakeResult(
                self.names,
                [
                    FakeBox([10, 20, 30, 40], 0.91, 1),
                    FakeBox([50, 60, 70, 80], 0.88, 0),
                ],
            )
        ]


class FakeYolov5Results:
    xyxy = [
        [
            [10, 20, 30, 40, 0.91, 1],
            [50, 60, 70, 80, 0.88, 0],
        ]
    ]


class FakeYolov5Model:
    names = {0: "forklift_1", 1: "forklift_2"}

    def __init__(self):
        self.calls = []

    def __call__(self, frame, size):
        self.calls.append({"frame": frame, "size": size})
        return FakeYolov5Results()


class WarningYolov5Model(FakeYolov5Model):
    def __call__(self, frame, size):
        warnings.warn(
            "`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return super().__call__(frame, size)


class ForkliftDetectorTest(unittest.TestCase):
    def test_detect_loads_model_with_configured_path_and_returns_forklift_detections(self):
        loaded_paths = []

        def loader(path):
            loaded_paths.append(path)
            return FakeModel()

        detector = ForkliftDetector(
            "models/forklift.pt",
            confidence=0.4,
            class_names=["forklift"],
            model_loader=loader,
        )

        detections = detector.detect(frame="frame-1")

        self.assertEqual(loaded_paths, ["models/forklift.pt"])
        self.assertEqual(
            detections,
            [
                {
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "score": 0.91,
                    "class_name": "forklift",
                }
            ],
        )
        self.assertEqual(detector.model.calls, [{"frame": "frame-1", "conf": 0.4, "verbose": False}])

    def test_detect_returns_configured_class_from_yolov5_results(self):
        detector = ForkliftDetector(
            "models/best.pt",
            confidence=0.4,
            class_names=["forklift_2"],
            model_loader=lambda path: FakeYolov5Model(),
        )

        detections = detector.detect(frame="frame-1")

        self.assertEqual(
            detections,
            [
                {
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "score": 0.91,
                    "class_name": "forklift_2",
                }
            ],
        )
        self.assertEqual(detector.model.calls, [{"frame": "frame-1", "size": 640}])

    def test_detect_suppresses_known_yolov5_autocast_future_warning(self):
        detector = ForkliftDetector(
            "models/best.pt",
            confidence=0.4,
            class_names=["forklift_2"],
            model_loader=lambda path: WarningYolov5Model(),
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            detections = detector.detect(frame="frame-1")

        self.assertEqual(
            detections,
            [
                {
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "score": 0.91,
                    "class_name": "forklift_2",
                }
            ],
        )
        self.assertFalse(
            [
                warning
                for warning in captured_warnings
                if issubclass(warning.category, FutureWarning)
                and "torch.cuda.amp.autocast(args...)" in str(warning.message)
            ]
        )


if __name__ == "__main__":
    unittest.main()
