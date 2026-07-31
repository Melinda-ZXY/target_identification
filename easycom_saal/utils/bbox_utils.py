from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def bbox_to_xyxy(bbox: Any) -> tuple[float, float, float, float]:
    if hasattr(bbox, "x1"):
        return float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)
    if isinstance(bbox, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(bbox):
            return float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
        if {"left", "top", "right", "bottom"}.issubset(bbox):
            return (
                float(bbox["left"]),
                float(bbox["top"]),
                float(bbox["right"]),
                float(bbox["bottom"]),
            )
        if {"x", "y", "w", "h"}.issubset(bbox):
            x = float(bbox["x"])
            y = float(bbox["y"])
            return x, y, x + float(bbox["w"]), y + float(bbox["h"])
        if {"x", "y", "width", "height"}.issubset(bbox):
            x = float(bbox["x"])
            y = float(bbox["y"])
            return x, y, x + float(bbox["width"]), y + float(bbox["height"])
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    raise ValueError(f"Cannot convert bbox to xyxy: {bbox!r}")


def bbox_center(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_area(bbox: Any) -> float:
    width, height = bbox_size(bbox)
    return width * height


def bbox_is_visible(bbox: Any) -> bool:
    if bbox is None:
        return False
    if hasattr(bbox, "visible"):
        if not bool(bbox.visible):
            return False
    if isinstance(bbox, dict):
        visible = bbox.get("visible", bbox.get("is_visible", True))
        if not bool(visible):
            return False
    return bbox_area(bbox) > 0.0


def bbox_inside_frame(bbox: Any, video_size: tuple[int, int], require_center: bool = True) -> bool:
    height, width = video_size
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    if require_center:
        cx, cy = bbox_center(bbox)
        return 0.0 <= cx < width and 0.0 <= cy < height
    return x2 > 0.0 and y2 > 0.0 and x1 < width and y1 < height


def scale_bbox(
    bbox: Any,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    from_h, from_w = from_size
    to_h, to_w = to_size
    x_scale = to_w / float(from_w)
    y_scale = to_h / float(from_h)
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    return x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale


def clamp_bbox(bbox: Any, video_size: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = video_size
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    return (
        min(max(x1, 0.0), float(width - 1)),
        min(max(y1, 0.0), float(height - 1)),
        min(max(x2, 0.0), float(width - 1)),
        min(max(y2, 0.0), float(height - 1)),
    )

