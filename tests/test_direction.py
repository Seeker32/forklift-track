import unittest

from src.direction import DirectionDetector


class DirectionDetectorTest(unittest.TestCase):
    def test_direction_detector_emits_in_event_when_track_crosses_with_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            line_width=4.0,
        )

        self.assertIsNone(detector.update({"track_id": 12, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]}))
        self.assertIsNone(detector.update({"track_id": 12, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]}))
        event = detector.update({"track_id": 12, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]})

        self.assertIsNotNone(event)
        self.assertEqual(event["camera_id"], "gate_01")
        self.assertEqual(event["track_id"], 12)
        self.assertEqual(event["direction"], "in")
        self.assertEqual(event["bbox"], [4, 4, 6, 6])

    def test_direction_detector_emits_out_event_when_track_crosses_against_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            line_width=4.0,
        )

        detector.update({"track_id": 7, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]})
        detector.update({"track_id": 7, "center": (5.0, 1.0), "bbox": [4, -1, 6, 1]})
        event = detector.update({"track_id": 7, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]})

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "out")

    def test_direction_detector_does_not_emit_duplicate_event_for_same_track_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            line_width=4.0,
        )

        detector.update({"track_id": 3, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]})
        detector.update({"track_id": 3, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]})
        first = detector.update({"track_id": 3, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]})
        detector.update({"track_id": 3, "center": (5.0, 1.0), "bbox": [4, -1, 6, 1]})
        detector.update({"track_id": 3, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]})
        detector.update({"track_id": 3, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]})
        duplicate = detector.update({"track_id": 3, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]})

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)

    def test_direction_detector_allows_opposite_direction_once_for_same_track(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            line_width=4.0,
        )

        detector.update({"track_id": 9, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]})
        detector.update({"track_id": 9, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]})
        first = detector.update({"track_id": 9, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]})
        detector.update({"track_id": 9, "center": (5.0, 1.0), "bbox": [4, -1, 6, 1]})
        second = detector.update({"track_id": 9, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]})

        self.assertEqual(first["direction"], "in")
        self.assertEqual(second["direction"], "out")

    def test_direction_detector_cleans_missing_track_state_after_threshold(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            max_missing_frames=30,
            line_width=4.0,
        )

        detector.update({"track_id": 4, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]}, frame_index=1)
        detector.update({"track_id": 4, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]}, frame_index=2)
        first = detector.update({"track_id": 4, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]}, frame_index=3)
        detector.prune_missing(frame_index=34)
        detector.update({"track_id": 4, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]}, frame_index=35)
        detector.update({"track_id": 4, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]}, frame_index=36)
        second = detector.update({"track_id": 4, "center": (5.0, 6.0), "bbox": [4, 4, 6, 6]}, frame_index=37)

        self.assertEqual(first["direction"], "in")
        self.assertEqual(second["direction"], "in")

    def test_direction_detector_does_not_emit_when_track_returns_to_entry_side(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            line_width=4.0,
        )

        self.assertIsNone(detector.update({"track_id": 2, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]}))
        self.assertIsNone(detector.update({"track_id": 2, "center": (5.0, -1.0), "bbox": [4, -3, 6, -1]}))
        self.assertIsNone(detector.update({"track_id": 2, "center": (5.0, -6.0), "bbox": [4, -8, 6, -6]}))


if __name__ == "__main__":
    unittest.main()
