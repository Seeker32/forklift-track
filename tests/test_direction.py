import unittest

from src.direction import DirectionDetector


class DirectionDetectorTest(unittest.TestCase):
    def test_direction_detector_emits_in_event_when_segment_crosses_with_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        self.assertIsNone(detector.update({"track_id": 12, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]}))
        event = detector.update({"track_id": 12, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})

        self.assertIsNotNone(event)
        self.assertEqual(event["camera_id"], "gate_01")
        self.assertEqual(event["track_id"], 12)
        self.assertEqual(event["direction"], "in")
        self.assertEqual(event["bbox"], [3, -3, 7, 3])

    def test_direction_detector_emits_out_event_when_segment_crosses_against_in_direction(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 7, "center": (5.0, 5.0), "bbox": [3, 2, 7, 8]})
        event = detector.update({"track_id": 7, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "out")

    def test_direction_detector_does_not_emit_duplicate_event_while_bbox_straddles_line(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 3, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]})
        first = detector.update({"track_id": 3, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})
        # Bbox still straddles the line — same crossing, should NOT emit a duplicate
        second = detector.update({"track_id": 3, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})

        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_direction_detector_allows_opposite_direction_once_for_same_track(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 9, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]})
        first = detector.update({"track_id": 9, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})
        detector.update({"track_id": 9, "center": (5.0, 5.0), "bbox": [3, 2, 7, 8]})
        second = detector.update({"track_id": 9, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})

        self.assertEqual(first["direction"], "in")
        self.assertEqual(second["direction"], "out")

    def test_direction_detector_cleans_missing_track_state_after_threshold(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
            max_missing_frames=30,
        )

        detector.update({"track_id": 4, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]}, frame_index=1)
        first = detector.update({"track_id": 4, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]}, frame_index=2)
        detector.prune_missing(frame_index=33)
        detector.update({"track_id": 4, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]}, frame_index=34)
        second = detector.update({"track_id": 4, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]}, frame_index=35)

        self.assertEqual(first["direction"], "in")
        self.assertEqual(second["direction"], "in")

    def test_direction_detector_records_debug_snapshot_for_last_track_update(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 2, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]}, frame_index=7)

        self.assertEqual(
            detector.last_debug_snapshot,
            {
                "frame": 7,
                "track_id": 2,
                "bbox": [3, -8, 7, -2],
                "center": (5.0, -5.0),
                "bottom": (5.0, -2.0),
                "crossed": False,
                "class_name": None,
                "event": None,
            },
        )

    def test_forklift_empty_crossing_does_not_emit_event(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 1, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2], "class_name": "forklift_empty"})
        event = detector.update({"track_id": 1, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3], "class_name": "forklift_empty"})

        self.assertIsNone(event)

    def test_forklift_with_load_crossing_emits_event(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 2, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2], "class_name": "forklift_with_load"})
        event = detector.update({"track_id": 2, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3], "class_name": "forklift_with_load"})

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "in")
        self.assertEqual(event["class_name"], "forklift_with_load")

    def test_missing_class_name_still_emits_event(self):
        detector = DirectionDetector(
            camera_id="gate_01",
            line_start=(0.0, 0.0),
            line_end=(10.0, 0.0),
            in_direction=(0.0, 1.0),
        )

        detector.update({"track_id": 3, "center": (5.0, -5.0), "bbox": [3, -8, 7, -2]})
        event = detector.update({"track_id": 3, "center": (5.0, 0.0), "bbox": [3, -3, 7, 3]})

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "in")
        self.assertIsNone(event["class_name"])


if __name__ == "__main__":
    unittest.main()
