from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def point_side(point: Sequence[float], line_start: Sequence[float], line_end: Sequence[float]) -> float:
    """Return the signed side of a point relative to a directed line."""
    px, py = _point(point)
    ax, ay = _point(line_start)
    bx, by = _point(line_end)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


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
