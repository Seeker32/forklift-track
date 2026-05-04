from __future__ import annotations

from collections.abc import Sequence
from math import hypot

Point = tuple[float, float]


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def point_side(point: Sequence[float], line_start: Sequence[float], line_end: Sequence[float]) -> float:
    """Return the signed side of a point relative to a directed line."""
    px, py = _point(point)
    ax, ay = _point(line_start)
    bx, by = _point(line_end)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def signed_distance_to_line(point: Sequence[float], line_start: Sequence[float], line_end: Sequence[float]) -> float:
    """Return the signed perpendicular distance to a directed line."""
    ax, ay = _point(line_start)
    bx, by = _point(line_end)
    length = hypot(bx - ax, by - ay)
    if length == 0:
        raise ValueError("Line start and end must be different points")
    return point_side(point, line_start, line_end) / length


def point_projection_ratio(point: Sequence[float], line_start: Sequence[float], line_end: Sequence[float]) -> float:
    px, py = _point(point)
    ax, ay = _point(line_start)
    bx, by = _point(line_end)
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        raise ValueError("Line start and end must be different points")
    return ((px - ax) * dx + (py - ay) * dy) / length_squared


def point_in_line_zone(
    point: Sequence[float],
    line_start: Sequence[float],
    line_end: Sequence[float],
    line_width: float,
) -> bool:
    ratio = point_projection_ratio(point, line_start, line_end)
    return 0 <= ratio <= 1 and abs(signed_distance_to_line(point, line_start, line_end)) <= line_width / 2.0


def line_zone_side(
    point: Sequence[float],
    line_start: Sequence[float],
    line_end: Sequence[float],
    line_width: float,
) -> int:
    if point_in_line_zone(point, line_start, line_end, line_width):
        return 0
    return 1 if signed_distance_to_line(point, line_start, line_end) > 0 else -1


def movement_vector(previous: Sequence[float], current: Sequence[float]) -> Point:
    prev_x, prev_y = _point(previous)
    curr_x, curr_y = _point(current)
    return curr_x - prev_x, curr_y - prev_y


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = _point(a)
    bx, by = _point(b)
    return ax * bx + ay * by


def crossed_line(
    previous: Sequence[float],
    current: Sequence[float],
    line_start: Sequence[float],
    line_end: Sequence[float],
) -> bool:
    """Return True when a movement segment intersects the configured line segment."""
    prev_side = point_side(previous, line_start, line_end)
    curr_side = point_side(current, line_start, line_end)

    if prev_side == 0 and curr_side == 0:
        return False
    if (prev_side > 0 and curr_side > 0) or (prev_side < 0 and curr_side < 0):
        return False

    line_prev_side = point_side(line_start, previous, current)
    line_curr_side = point_side(line_end, previous, current)
    if line_prev_side == 0 or line_curr_side == 0:
        return True
    return (line_prev_side > 0) != (line_curr_side > 0)
