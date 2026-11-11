from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class TextRegion:
    box: Box
    source_text: str = ""
    translated_text: str = ""
    confidence: float | None = None
    orientation: str = "unknown"
    bubble_box: Box | None = None


def clamp(box, width, height):
    x1, y1, x2, y2 = box
    return (max(0, min(width, x1)), max(0, min(height, y1)),
            max(0, min(width, x2)), max(0, min(height, y2)))


def polygon_box(points, width, height, scale_x=1, scale_y=1):
    xs, ys = zip(*points)
    return clamp((floor(min(xs) / scale_x), floor(min(ys) / scale_y),
                  ceil(max(xs) / scale_x), ceil(max(ys) / scale_y)), width, height)


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]),
                 min(a[2], b[2]), min(a[3], b[3])))


def orientation(box):
    width, height = box[2] - box[0], box[3] - box[1]
    if height > width * 1.25:
        return "vertical"
    if width > height * 1.25:
        return "horizontal"
    return "unknown"


def group_regions(rgb, boxes, min_side=8):
    height, width = rgb.shape[:2]
    candidates = []
    for box in sorted(boxes, key=area, reverse=True):
        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < min_side or area(box) >= width * height * .85:
            continue
        if any(intersection(box, other) >= .65 * area(box) for other in candidates):
            continue
        candidates.append(box)
    if not candidates:
        return []

    # A shared enclosed bright component is stronger evidence than line proximity.
    light = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) >= 215).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    groups = {}
    regions = []
    for box in candidates:
        x1, y1, x2, y2 = box
        values, counts = np.unique(labels[y1:y2, x1:x2], return_counts=True)
        bright = values != 0
        if bright.any():
            label = values[bright][counts[bright].argmax()]
            x, y, w, h, _ = stats[label]
            if (0 < x <= x1 < x2 <= x + w < width
                    and 0 < y <= y1 < y2 <= y + h < height
                    and counts[bright].max() >= area(box) * .25):
                groups.setdefault(int(label), []).append(box)
                continue
        regions.append(TextRegion(box, orientation=orientation(box)))

    for lines in groups.values():
        box = (min(b[0] for b in lines), min(b[1] for b in lines),
               max(b[2] for b in lines), max(b[3] for b in lines))
        directions = {orientation(b) for b in lines} - {"unknown"}
        if len(directions) > 1 or area(box) >= width * height * .85:
            regions.extend(TextRegion(b, orientation=orientation(b)) for b in lines)
            continue
        # Limit fitting to the text span since an oval's corners can contain artwork.
        regions.append(TextRegion(box, orientation=next(iter(directions), orientation(box)),
                                  bubble_box=box))
    return sorted(regions, key=lambda r: (r.box[1], r.box[0]))


class JapaneseDetector:
    def __init__(self, engine, min_side):
        self.engine = engine
        self.min_side = min_side

    def detect(self, rgb):
        height, width = rgb.shape[:2]
        polygons, _ = self.engine(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  use_det=True, use_cls=False, use_rec=False)
        boxes = [] if polygons is None else [
            polygon_box(points, width, height) for points in polygons
        ]
        return group_regions(rgb, boxes, self.min_side)
