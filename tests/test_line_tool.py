import unittest

from src.line_tool import _collect_points, _line_zone_polygon, format_line_config


class LineToolTest(unittest.TestCase):
    def test_format_line_config_outputs_yaml_snippet_with_clicked_points(self):
        snippet = format_line_config([(300, 420), (980, 420)], in_direction=(0, -1), line_width=40)

        self.assertEqual(
            snippet,
            "line:\n"
            "  start: [300, 420]\n"
            "  end: [980, 420]\n"
            "line_width: 40\n"
            "in_direction: [0, -1]",
        )

    def test_line_zone_polygon_expands_line_by_half_width_on_both_sides(self):
        polygon = _line_zone_polygon((0, 0), (10, 0), line_width=4)

        self.assertEqual(polygon, [(0, 2), (10, 2), (10, -2), (0, -2)])

    def test_collect_points_waits_for_confirmation_after_second_point(self):
        cv2 = FakeCv2(keys=[0, ord(" ")])
        frame = FakeFrame()

        points = _collect_points(frame, cv2, line_width=4)

        self.assertEqual(points, [(1, 2), (5, 6)])
        self.assertEqual(cv2.wait_key_calls, 2)
        self.assertEqual(cv2.line_calls[0], ("frame-copy", (0, 3), (4, 7), (255, 255, 0), 1))
        self.assertEqual(cv2.line_calls[-1], ("frame-copy", (1, 2), (5, 6), (0, 255, 0), 2))


class FakeFrame:
    def copy(self):
        return "frame-copy"


class FakeCv2:
    EVENT_LBUTTONDOWN = 1

    def __init__(self, keys):
        self.keys = list(keys)
        self.line_calls = []
        self.wait_key_calls = 0
        self.callback = None

    def namedWindow(self, window_name):
        self.window_name = window_name

    def setMouseCallback(self, window_name, callback):
        self.callback = callback

    def circle(self, preview, point, radius, color, thickness):
        pass

    def line(self, preview, start, end, color, thickness):
        self.line_calls.append((preview, start, end, color, thickness))

    def imshow(self, window_name, preview):
        pass

    def waitKey(self, delay):
        self.wait_key_calls += 1
        if self.wait_key_calls == 1:
            self.callback(self.EVENT_LBUTTONDOWN, 1, 2, None, None)
            self.callback(self.EVENT_LBUTTONDOWN, 5, 6, None, None)
        return self.keys.pop(0)

    def destroyWindow(self, window_name):
        pass


if __name__ == "__main__":
    unittest.main()
