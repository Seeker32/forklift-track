import tempfile
import unittest
from pathlib import Path

from src.pipeline import load_config, run


class PipelineTest(unittest.TestCase):
    def test_run_processes_frames_and_collects_direction_events(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: fake-source
    line:
      start: [0, 0]
      end: [10, 0]
    line_width: 4
    in_direction: [0, 1]
"""
        )
        captures = {"fake-source": FakeCapture(["frame-1", "frame-2", "frame-3"])}
        events = []
        detectors = []
        trackers = []

        run(
            config_path,
            detector_factory=lambda **kwargs: detectors.append(FakeDetector()) or detectors[-1],
            tracker_factory=lambda: trackers.append(FakeTracker()) or trackers[-1],
            capture_factory=lambda source: captures[source],
            event_sink=events.append,
        )

        self.assertEqual(detectors[0].frames, ["frame-1", "frame-2", "frame-3"])
        self.assertEqual(trackers[0].detections, [[{"bbox": [0, 0, 2, 2], "score": 0.9, "class_name": "forklift_2"}]] * 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["camera_id"], "gate_01")
        self.assertEqual(events[0]["track_id"], 5)
        self.assertEqual(events[0]["direction"], "in")
        self.assertTrue(captures["fake-source"].released)

    def test_run_raises_clear_error_when_video_source_cannot_open(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: missing-source
    line:
      start: [0, 0]
      end: [10, 0]
    in_direction: [0, 1]
"""
        )

        with self.assertRaisesRegex(RuntimeError, "gate_01.*missing-source"):
            run(
                config_path,
                detector_factory=lambda **kwargs: FakeDetector(),
                tracker_factory=lambda: FakeTracker(),
                capture_factory=lambda source: FakeCapture([], opened=False),
                event_sink=lambda event: None,
            )

    def test_run_raises_value_error_for_missing_required_camera_fields(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: fake-source
"""
        )

        with self.assertRaisesRegex(ValueError, "line"):
            run(
                config_path,
                detector_factory=lambda **kwargs: FakeDetector(),
                tracker_factory=lambda: FakeTracker(),
                capture_factory=lambda source: FakeCapture([]),
                event_sink=lambda event: None,
            )

    def test_run_processes_multiple_cameras_independently(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: source-1
    line:
      start: [0, 0]
      end: [10, 0]
    line_width: 4
    in_direction: [0, 1]
  - camera_id: gate_02
    source: source-2
    line:
      start: [0, 0]
      end: [10, 0]
    line_width: 4
    in_direction: [0, 1]
"""
        )
        captures = {
            "source-1": FakeCapture(["a1", "a2", "a3"]),
            "source-2": FakeCapture(["b1", "b2", "b3"]),
        }
        events = []

        run(
            config_path,
            detector_factory=lambda **kwargs: FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: captures[source],
            event_sink=events.append,
        )

        self.assertEqual([event["camera_id"] for event in events], ["gate_01", "gate_02"])

    def test_load_config_defaults_line_width(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: fake-source
    line:
      start: [0, 0]
      end: [10, 0]
    in_direction: [0, 1]
"""
        )

        config = load_config(config_path)

        self.assertEqual(config["cameras"][0]["line_width"], 40.0)

    def test_load_config_normalizes_explicit_line_width(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: fake-source
    line:
      start: [0, 0]
      end: [10, 0]
    line_width: 60
    in_direction: [0, 1]
"""
        )

        config = load_config(config_path)

        self.assertEqual(config["cameras"][0]["line_width"], 60.0)

    def write_config(self, content):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "cameras.yaml"
        path.write_text(content, encoding="utf-8")
        return path


class FakeDetector:
    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return [{"bbox": [0, 0, 2, 2], "score": 0.9, "class_name": "forklift_2"}]


class FakeTracker:
    def __init__(self):
        self.detections = []
        self.frame_count = 0

    def update(self, detections):
        self.detections.append(detections)
        self.frame_count += 1
        bottom_y_by_frame = {1: -6, 2: -1, 3: 6}
        bottom_y = bottom_y_by_frame[self.frame_count]
        return [{"track_id": 5, "center": [5, bottom_y], "bbox": [4, bottom_y - 2, 6, bottom_y], "score": 0.9}]


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


if __name__ == "__main__":
    unittest.main()
