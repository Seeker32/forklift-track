import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import _create_detector, load_config, run


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
    class_name: "forklift_with_load"
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
        self.assertEqual(trackers[0].detections, [[{"bbox": [0, 0, 2, 2], "score": 0.9, "class_name": "forklift_with_load"}]] * 3)
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

    def test_run_model_path_override_replaces_camera_model_config(self):
        config_path = self.write_config(
            """
cameras:
  - camera_id: gate_01
    source: fake-source
    line:
      start: [0, 0]
      end: [10, 0]
    in_direction: [0, 1]
    model_path: models/best.pt
"""
        )
        captures = {"fake-source": FakeCapture([])}
        detector_kwargs = []

        run(
            config_path,
            model_path="models/inference_model.onnx",
            detector_factory=lambda **kwargs: detector_kwargs.append(kwargs) or FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: captures[source],
            event_sink=lambda event: None,
        )

        self.assertEqual(detector_kwargs[0]["model_path"], "models/inference_model.onnx")

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

    def test_run_debug_logs_frame_counts_track_zone_state_and_events(self):
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
    class_name: "forklift_with_load"
"""
        )
        captures = {"fake-source": FakeCapture(["frame-1", "frame-2", "frame-3"])}
        debug_lines = []

        run(
            config_path,
            detector_factory=lambda **kwargs: FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: captures[source],
            event_sink=lambda event: None,
            debug=True,
            debug_sink=debug_lines.append,
        )

        self.assertIn("[debug] camera=gate_01 frame=1 detections=1 tracks=1", debug_lines)
        self.assertIn(
            "[debug] camera=gate_01 frame=1 track_id=5 bbox=[4, -12, 6, -6] center=(5.0, -9.0) "
            "bottom=(5.0, -6.0) crossed=False event=None",
            debug_lines,
        )
        self.assertIn(
            "[debug] camera=gate_01 frame=2 track_id=5 bbox=[4, -5, 6, 1] center=(5.0, -2.0) "
            "bottom=(5.0, 1.0) crossed=True event=in",
            debug_lines,
        )

    def test_run_debug_video_writes_every_processed_frame_with_annotations(self):
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
    class_name: "forklift_with_load"
"""
        )
        captures = {"fake-source": FakeCapture(["frame-1", "frame-2", "frame-3"])}
        writers = []

        run(
            config_path,
            detector_factory=lambda **kwargs: FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: captures[source],
            event_sink=lambda event: None,
            debug_video=True,
            debug_video_writer_factory=lambda **kwargs: writers.append(FakeDebugVideoWriter(**kwargs)) or writers[-1],
        )

        self.assertEqual(len(writers), 1)
        self.assertEqual(writers[0].camera_config["camera_id"], "gate_01")
        self.assertEqual(writers[0].first_frame, "frame-1")
        self.assertEqual([write["frame"] for write in writers[0].writes], ["frame-1", "frame-2", "frame-3"])
        self.assertEqual([write["frame_index"] for write in writers[0].writes], [1, 2, 3])
        self.assertEqual(writers[0].writes[0]["detections"], [{"bbox": [0, 0, 2, 2], "score": 0.9, "class_name": "forklift_with_load"}])
        self.assertEqual(writers[0].writes[0]["tracks"][0]["track_id"], 5)
        self.assertEqual(writers[0].writes[1]["events"][0]["direction"], "in")
        self.assertTrue(writers[0].released)

    def test_run_debug_video_creates_separate_writers_for_multiple_cameras(self):
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
            "source-1": FakeCapture(["a1"]),
            "source-2": FakeCapture(["b1"]),
        }
        writers = []

        run(
            config_path,
            detector_factory=lambda **kwargs: FakeDetector(),
            tracker_factory=lambda: FakeTracker(),
            capture_factory=lambda source: captures[source],
            event_sink=lambda event: None,
            debug_video=True,
            debug_video_writer_factory=lambda **kwargs: writers.append(FakeDebugVideoWriter(**kwargs)) or writers[-1],
        )

        self.assertEqual([writer.camera_config["camera_id"] for writer in writers], ["gate_01", "gate_02"])
        self.assertTrue(all(writer.released for writer in writers))

    def test_run_debug_video_does_not_create_writer_when_video_source_cannot_open(self):
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
        writers = []

        with self.assertRaisesRegex(RuntimeError, "gate_01.*missing-source"):
            run(
                config_path,
                detector_factory=lambda **kwargs: FakeDetector(),
                tracker_factory=lambda: FakeTracker(),
                capture_factory=lambda source: FakeCapture([], opened=False),
                event_sink=lambda event: None,
                debug_video=True,
                debug_video_writer_factory=lambda **kwargs: writers.append(FakeDebugVideoWriter(**kwargs))
                or writers[-1],
            )

        self.assertEqual(writers, [])

    def test_create_detector_uses_onnx_detector_for_onnx_model(self):
        with patch("src.pipeline.ONNXForkliftDetector", create=True) as onnx_detector, patch(
            "src.pipeline.ForkliftDetector"
        ) as yolo_detector:
            detector = _create_detector(
                model_path="models/inference_model.onnx",
                confidence=0.3,
                class_name="forklift_2",
            )

        self.assertEqual(detector, onnx_detector.return_value)
        onnx_detector.assert_called_once_with(onnx_path="models/inference_model.onnx", confidence=0.3)
        yolo_detector.assert_not_called()

    def test_create_detector_treats_onnx_suffix_case_insensitively(self):
        with patch("src.pipeline.ONNXForkliftDetector", create=True) as onnx_detector:
            detector = _create_detector(
                model_path="models/INFERENCE_MODEL.ONNX",
                confidence=0.3,
                class_name="forklift_2",
            )

        self.assertEqual(detector, onnx_detector.return_value)
        onnx_detector.assert_called_once_with(onnx_path="models/INFERENCE_MODEL.ONNX", confidence=0.3)

    def test_create_detector_uses_yolo_detector_for_non_onnx_model(self):
        with patch("src.pipeline.ForkliftDetector") as yolo_detector:
            detector = _create_detector(
                model_path="models/best.pt",
                confidence=0.4,
                class_name="forklift_2",
            )

        self.assertEqual(detector, yolo_detector.return_value)
        yolo_detector.assert_called_once_with(
            model_path="models/best.pt",
            confidence=0.4,
            class_name="forklift_2",
        )

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
        return [{"bbox": [0, 0, 2, 2], "score": 0.9, "class_name": "forklift_with_load"}]


class FakeTracker:
    def __init__(self):
        self.detections = []
        self.frame_count = 0

    def update(self, detections):
        self.detections.append(detections)
        self.frame_count += 1
        bottom_y_by_frame = {1: -6, 2: 1, 3: 6}
        bottom_y = bottom_y_by_frame[self.frame_count]
        return [{"track_id": 5, "center": [5, bottom_y - 3], "bbox": [4, bottom_y - 6, 6, bottom_y], "score": 0.9}]


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


class FakeDebugVideoWriter:
    def __init__(self, *, camera_config, capture, first_frame):
        self.camera_config = camera_config
        self.capture = capture
        self.first_frame = first_frame
        self.writes = []
        self.released = False

    def write(self, *, frame, frame_index, detections, tracks, events):
        self.writes.append(
            {
                "frame": frame,
                "frame_index": frame_index,
                "detections": detections,
                "tracks": tracks,
                "events": events,
            }
        )

    def release(self):
        self.released = True


if __name__ == "__main__":
    unittest.main()
