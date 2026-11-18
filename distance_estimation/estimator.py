"""
Monocular distance estimation with the pinhole-camera model  D = W_r * f / w_b.

Per detection this module adds:
    width_prior_cm  W_r actually used (frontal or side-profile, Table IV)
    width_view      "front" | "side"
    bbox_width_ud   w_b after lens-distortion correction (capture-frame pixels)
    distance_raw    single-frame pinhole estimate (m)
    clipped         box touches the left/right frame border -> w_b truncated
    clipped_both    box touches both borders (object wider than the view)
    min_measurable  smallest distance at which the whole prior width fits (m)
    occluded        overlaps a foreground box with IoU > OCCLUSION_IOU

finalize() (called after tracking/smoothing) adds danger_level and position.
"""

import numpy as np
import cv2

from config import (CLIPPED_DANGER_FRAC, DISTANCE_CAUTION, DISTANCE_DANGER,
                    DISTANCE_WARNING, EDGE_MARGIN_PX, KNOWN_WIDTHS,
                    OCCLUSION_IOU, SECTOR_SPLITS, SIDE_PROFILE)
from distance_estimation.calibration import load_intrinsics

LEVELS = ("DANGER", "WARNING", "CAUTION", "SAFE")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


def danger_level(dist_m):
    """Table III."""
    if dist_m <= DISTANCE_DANGER:
        return "DANGER"
    if dist_m <= DISTANCE_WARNING:
        return "WARNING"
    if dist_m <= DISTANCE_CAUTION:
        return "CAUTION"
    return "SAFE"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_width(cls, bbox_w, bbox_h):
    """Frontal vs side-profile prior from the box aspect ratio (Table IV)."""
    front = KNOWN_WIDTHS.get(cls, KNOWN_WIDTHS["default"])
    side = SIDE_PROFILE.get(cls)
    if side and bbox_h > 0 and (bbox_w / bbox_h) > side[1]:
        return side[0], "side"
    return front, "front"


class Estimator:
    def __init__(self, frame_width=None, frame_height=None):
        self.intr = None
        if frame_width and frame_height:
            self._load(frame_width, frame_height)

    def _load(self, w, h):
        self.intr = load_intrinsics(w, h)
        print(f"[Estimator] {self.intr.describe()}")

    @property
    def focal(self):
        return self.intr.fx

    # ── distortion-corrected box width ───────────────────────────────────────
    def _undistorted_width(self, x1, x2, cy):
        if self.intr.dist is None or not np.any(self.intr.dist):
            return float(x2 - x1)
        pts = np.array([[[x1, cy]], [[x2, cy]]], dtype=np.float64)
        ud = cv2.undistortPoints(pts, self.intr.K, self.intr.dist, P=self.intr.K)
        return float(abs(ud[1, 0, 0] - ud[0, 0, 0]))

    def pinhole(self, width_cm, width_px):
        if width_px <= 0:
            return 999.0
        return (width_cm * self.focal) / width_px / 100.0

    # ── per-frame raw estimate ───────────────────────────────────────────────
    def estimate(self, detections, frame_width, frame_height):
        if self.intr is None or (self.intr.width, self.intr.height) != (frame_width, frame_height):
            self._load(frame_width, frame_height)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            bw, bh = d["bbox_width"], d["bbox_height"]
            w_cm, view = select_width(d["class"], bw, bh)
            w_px = self._undistorted_width(x1, x2, (y1 + y2) / 2)

            left = x1 <= EDGE_MARGIN_PX
            right = x2 >= frame_width - 1 - EDGE_MARGIN_PX

            d["width_prior_cm"] = w_cm
            d["width_view"] = view
            d["bbox_width_ud"] = round(w_px, 1)
            d["distance_raw"] = round(self.pinhole(w_cm, w_px), 3)
            d["clipped"] = left or right
            d["clipped_both"] = left and right
            d["min_measurable"] = round(self.pinhole(w_cm, frame_width), 3)
            d["occluded"] = False

        # Occlusion: the box whose bottom edge is lower is taken as foreground.
        for i, a in enumerate(detections):
            for b in detections[i + 1:]:
                if iou(a["bbox"], b["bbox"]) > OCCLUSION_IOU:
                    back = a if a["bbox"][3] < b["bbox"][3] else b
                    back["occluded"] = True
        return detections

    # ── after smoothing ──────────────────────────────────────────────────────
    def finalize(self, detections, frame_width):
        """Assign danger level (with near-field clipping rule) and LEFT/CENTER/RIGHT."""
        for d in detections:
            dist = d.get("distance", d["distance_raw"])
            level = danger_level(dist)
            # A clipped box only gives an UPPER bound on distance; the object is
            # at least this close. Escalate when the box dominates the view.
            if d["clipped_both"] or (d["clipped"] and d["bbox_width"] >= CLIPPED_DANGER_FRAC * frame_width):
                level = "DANGER"
            d["danger_level"] = level
            d["position"] = sector_of_point((d["bbox"][0] + d["bbox"][2]) / 2, frame_width)
        return detections


def sector_of_point(x, frame_width):
    a, b = SECTOR_SPLITS
    if x < a * frame_width:
        return "LEFT"
    if x <= b * frame_width:
        return "CENTER"
    return "RIGHT"
