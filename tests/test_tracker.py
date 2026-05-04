import unittest

from src.tracker import ByteTrackTracker


class ByteTrackTrackerTest(unittest.TestCase):
    def test_update_returns_normalized_tracking_output(self):
        tracker = ByteTrackTracker(
            tracker_backend=FakeBackend(
                [
                    [
                        [10.0, 20.0, 30.0, 50.0, 12.0, 0.86, 0.0, 0.0],
                    ]
                ]
            )
        )

        tracks = tracker.update(
            [
                {
                    "bbox": [10, 20, 30, 50],
                    "score": 0.86,
                    "class_name": "forklift_2",
                }
            ]
        )

        self.assertEqual(
            tracks,
            [
                {
                    "track_id": 12,
                    "bbox": [10.0, 20.0, 30.0, 50.0],
                    "center": [20.0, 35.0],
                    "score": 0.86,
                }
            ],
        )

    def test_update_returns_empty_list_for_empty_detections(self):
        tracker = ByteTrackTracker(tracker_backend=FakeBackend([[]]))

        self.assertEqual(tracker.update([]), [])
        self.assertEqual(len(tracker.tracker_backend.received_results), 1)
        self.assertEqual(len(tracker.tracker_backend.received_results[0]), 0)

    def test_update_keeps_same_track_id_across_consecutive_frames(self):
        tracker = ByteTrackTracker(
            tracker_backend=FakeBackend(
                [
                    [[10.0, 20.0, 30.0, 50.0, 5.0, 0.9, 0.0, 0.0]],
                    [[12.0, 20.0, 32.0, 50.0, 5.0, 0.88, 0.0, 0.0]],
                ]
            )
        )

        first = tracker.update([{"bbox": [10, 20, 30, 50], "score": 0.9, "class_name": "forklift_2"}])
        second = tracker.update([{"bbox": [12, 20, 32, 50], "score": 0.88, "class_name": "forklift_2"}])

        self.assertEqual(first[0]["track_id"], 5)
        self.assertEqual(second[0]["track_id"], 5)

    def test_real_bytetrack_backend_keeps_same_track_id_for_nearby_detections(self):
        tracker = ByteTrackTracker()

        first = tracker.update([{"bbox": [10, 20, 30, 50], "score": 0.9, "class_name": "forklift_2"}])
        second = tracker.update([{"bbox": [12, 20, 32, 50], "score": 0.88, "class_name": "forklift_2"}])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])


class FakeBackend:
    def __init__(self, updates):
        self.updates = list(updates)
        self.received_results = []

    def update(self, results):
        self.received_results.append(results)
        return self.updates.pop(0)


if __name__ == "__main__":
    unittest.main()
