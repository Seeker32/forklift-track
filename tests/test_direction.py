import unittest

from src.direction import DirectionDetector


class DirectionDetectorTest(unittest.TestCase):
    def test_direction_detector_emits_in_event_when_track_crosses_with_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        self.assertIsNone(detector.update({"track_id": 12, "center": (5.0, -2.0), "bbox": [1, 1, 2, 2]}))
        event = detector.update({"track_id": 12, "center": (5.0, 2.0), "bbox": [1, 1, 2, 2]})

        self.assertIsNotNone(event)
        self.assertEqual(event["camera_id"], "gate_01")
        self.assertEqual(event["track_id"], 12)
        self.assertEqual(event["direction"], "in")
        self.assertEqual(event["bbox"], [1, 1, 2, 2])

    def test_direction_detector_emits_out_event_when_track_crosses_against_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 7, "center": (5.0, 2.0), "bbox": [1, 1, 2, 2]})
        event = detector.update({"track_id": 7, "center": (5.0, -2.0), "bbox": [1, 1, 2, 2]})

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "out")

    def test_direction_detector_does_not_emit_duplicate_event_for_same_track_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 3, "center": (5.0, -2.0), "bbox": [1, 1, 2, 2]})
        first = detector.update({"track_id": 3, "center": (5.0, 2.0), "bbox": [1, 1, 2, 2]})
        detector.update({"track_id": 3, "center": (5.0, -2.0), "bbox": [1, 1, 2, 2]})
        duplicate = detector.update({"track_id": 3, "center": (5.0, 2.0), "bbox": [1, 1, 2, 2]})

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)


if __name__ == "__main__":
    unittest.main()
