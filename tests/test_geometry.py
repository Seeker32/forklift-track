import unittest

from src.geometry import crossed_line, movement_vector, point_in_line_zone, point_side


class GeometryTest(unittest.TestCase):
    def test_point_side_distinguishes_points_on_opposite_sides_of_line(self):
        line_start = (0.0, 0.0)
        line_end = (10.0, 0.0)

        self.assertGreater(point_side((5.0, 5.0), line_start, line_end), 0)
        self.assertLess(point_side((5.0, -5.0), line_start, line_end), 0)
        self.assertEqual(point_side((5.0, 0.0), line_start, line_end), 0)

    def test_crossed_line_detects_track_segment_intersection(self):
        line_start = (0.0, 0.0)
        line_end = (10.0, 0.0)

        self.assertTrue(crossed_line((5.0, -2.0), (5.0, 2.0), line_start, line_end))
        self.assertFalse(crossed_line((1.0, -2.0), (2.0, -1.0), line_start, line_end))

    def test_crossed_line_ignores_intersections_outside_line_segment(self):
        line_start = (0.0, 0.0)
        line_end = (10.0, 0.0)

        self.assertFalse(crossed_line((15.0, -2.0), (15.0, 2.0), line_start, line_end))

    def test_crossed_line_ignores_movement_along_line(self):
        line_start = (0.0, 0.0)
        line_end = (10.0, 0.0)

        self.assertFalse(crossed_line((2.0, 0.0), (8.0, 0.0), line_start, line_end))

    def test_movement_vector_uses_previous_and_current_point(self):
        self.assertEqual(movement_vector((3.0, 4.0), (8.0, 1.0)), (5.0, -3.0))

    def test_point_in_line_zone_checks_width_and_segment_bounds(self):
        line_start = (0.0, 0.0)
        line_end = (10.0, 0.0)
        line_width = 4.0

        self.assertTrue(point_in_line_zone((5.0, 0.0), line_start, line_end, line_width))
        self.assertTrue(point_in_line_zone((5.0, 1.9), line_start, line_end, line_width))
        self.assertTrue(point_in_line_zone((5.0, -1.9), line_start, line_end, line_width))
        self.assertFalse(point_in_line_zone((5.0, 2.1), line_start, line_end, line_width))
        self.assertFalse(point_in_line_zone((12.0, 0.0), line_start, line_end, line_width))


if __name__ == "__main__":
    unittest.main()
