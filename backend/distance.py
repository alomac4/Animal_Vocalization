"""Uncalibrated monocular apparent-size range estimate, NOT measured depth.

26mm-equivalent preset maps the frame's long side to a hypothetical 36mm sensor
long side. Video crop, stabilization, zoom, pose and real dog size are unknown.
Physical height means the entire detection's projected vertical extent, NOT
withers/shoulder height. No tracking, identity or breed decision uses this value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

METRES_TO_FEET = 1 / .3048


@dataclass(frozen=True)
class DistancePreset:
    focal_equivalent_mm: float = 26.0
    dog_extent_m: float = .65

    def __post_init__(self):
        if (not math.isfinite(self.focal_equivalent_mm) or not 10 <= self.focal_equivalent_mm <= 200
                or not math.isfinite(self.dog_extent_m) or not .1 <= self.dog_extent_m <= 2):
            raise ValueError("Distance preset requires 10–200mm equivalent and 0.1–2m full-box dog extent")

    def estimate(self, box, width, height):
        if (len(box) != 4 or not all(math.isfinite(x) for x in box)
                or not math.isfinite(width) or not math.isfinite(height) or min(width,height) < 16):
            return None
        x1,y1,x2,y2 = box
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None
        # Tiny boxes and likely partial/edge-truncated animals have no usable size.
        margin = max(1., min(width,height) * .005)
        if x1 <= margin or y1 <= margin or x2 >= width-margin or y2 >= height-margin or y2-y1 < 12:
            return None
        focal_pixels = max(width,height) * self.focal_equivalent_mm / 36.0
        axial = focal_pixels * self.dog_extent_m / (y2-y1)
        # Approximate straight-line range to the box center, not ground distance.
        u,v = (x1+x2)/2-width/2, (y1+y2)/2-height/2
        metres = axial * math.sqrt(1 + (u/focal_pixels)**2 + (v/focal_pixels)**2)
        if not .1 <= metres <= 200:
            return None
        return {"metres": metres, "feet": metres * METRES_TO_FEET,
                "axial_metres": axial, "approximate": True}

    def metadata(self):
        return {"method": "uncalibrated_pinhole_box_extent", "approximate": True,
                "focal_equivalent_mm": self.focal_equivalent_mm, "dog_extent_m": self.dog_extent_m,
                "reference_sensor_long_side_mm": 36.0,
                "units": ["m", "ft"],
                "warning": "Assumed camera and dog size; not measured depth. Zoom, video crop, pose, occlusion and real size can cause large errors."}

    def notice(self):
        return (f"UNCALIBRATED distance estimate | assumed {self.focal_equivalent_mm:g}mm eq / "
                f"dog extent {self.dog_extent_m:g}m")


def format_distance(value):
    return (f"est. {value['metres']:.1f} m / {value['feet']:.1f} ft"
            if value is not None else "distance unavailable")