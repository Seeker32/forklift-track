from __future__ import annotations

from collections.abc import Callable, Sequence
from math import hypot
from typing import Any

Point = tuple[int, int]


def format_line_config(points: Sequence[Point], in_direction: Point = (0, -1), line_width: float = 40.0) -> str:
    if len(points) != 2:
        raise ValueError("Exactly two points are required to format a virtual line")

    start, end = points
    width = int(line_width) if float(line_width).is_integer() else float(line_width)
    return (
        "line:\n"
        f"  start: [{start[0]}, {start[1]}]\n"
        f"  end: [{end[0]}, {end[1]}]\n"
        f"line_width: {width}\n"
        f"in_direction: [{in_direction[0]}, {in_direction[1]}]"
    )


def select_line_from_source(
    source: str,
    *,
    line_width: float = 40.0,
    output: Callable[[str], None] = print,
) -> list[Point]:
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok:
        raise RuntimeError(f"Failed to read first frame from video source: {source}")

    points = _collect_points(frame, cv2, line_width=line_width)
    output(format_line_config(points, line_width=line_width))
    return points


def _collect_points(frame: Any, cv2_module: Any, *, line_width: float = 40.0) -> list[Point]:
    window_name = "Select virtual line - click 2 points, enter/space confirm, r reset, q/esc quit"
    points: list[Point] = []

    def redraw() -> None:
        preview = frame.copy()
        for point in points:
            cv2_module.circle(preview, point, 5, (0, 0, 255), -1)
        if len(points) == 2:
            _draw_line_zone(preview, cv2_module, points[0], points[1], line_width)
            cv2_module.line(preview, points[0], points[1], (0, 255, 0), 2)
        cv2_module.imshow(window_name, preview)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2_module.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        points.append((int(x), int(y)))
        redraw()

    cv2_module.namedWindow(window_name)
    cv2_module.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2_module.waitKey(20) & 0xFF
        if len(points) == 2 and key in (ord(" "), 10, 13):
            break
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (ord("q"), 27):
            break

    cv2_module.destroyWindow(window_name)
    if len(points) != 2:
        raise RuntimeError("Line selection cancelled before two points were selected")
    return points


def _line_zone_polygon(start: Point, end: Point, line_width: float) -> list[Point]:
    start_x, start_y = start
    end_x, end_y = end
    dx = end_x - start_x
    dy = end_y - start_y
    length = hypot(dx, dy)
    if length == 0:
        raise ValueError("Line start and end must be different points")

    half_width = line_width / 2.0
    normal_x = -dy / length
    normal_y = dx / length

    return [
        (round(start_x + normal_x * half_width), round(start_y + normal_y * half_width)),
        (round(end_x + normal_x * half_width), round(end_y + normal_y * half_width)),
        (round(end_x - normal_x * half_width), round(end_y - normal_y * half_width)),
        (round(start_x - normal_x * half_width), round(start_y - normal_y * half_width)),
    ]


def _draw_line_zone(preview: Any, cv2_module: Any, start: Point, end: Point, line_width: float) -> None:
    polygon = _line_zone_polygon(start, end, line_width)
    zone_color = (255, 255, 0)
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        cv2_module.line(preview, point, next_point, zone_color, 1)
