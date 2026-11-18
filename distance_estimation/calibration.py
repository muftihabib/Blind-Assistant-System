"""
Camera calibration.

Two procedures:

1. Checkerboard (full intrinsics, used for all reported results)
       python -m distance_estimation.calibration checkerboard --cols 9 --rows 6 --square-mm 25
   Captures views from the webcam at the configured FRAME_WIDTH x FRAME_HEIGHT
   (SPACE = capture, C = calibrate, Q = quit) or reads them from --images DIR,
   runs cv2.calibrateCamera and writes CALIBRATION_FILE.

2. Single-shot reference (quick field check only)
       python -m distance_estimation.calibration reference --width-cm 45 --distance-cm 200 --pixels 410
   f = (w_b * D) / W_r

IMPORTANT — coordinate frame: Ultralytics returns boxes in the ORIGINAL capture
frame (e.g. 1280x720), not in the 640x640 letterboxed network input. The focal
length used by the pinhole model must therefore be expressed at the capture
resolution. load_intrinsics() rescales the stored intrinsics if the runtime
resolution differs from the calibration resolution.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from config import (CALIBRATION_FILE, CAMERA_INDEX, FOCAL_LENGTH,
                    FRAME_HEIGHT, FRAME_WIDTH, NOMINAL_FOV_SOURCE)


class Intrinsics:
    """Camera matrix + distortion, valid at (width, height)."""

    def __init__(self, fx, fy, cx, cy, dist, width, height, source, rms=None):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1) if dist is not None else None
        self.width, self.height = int(width), int(height)
        self.source = source
        self.rms = rms

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def hfov_deg(self):
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    def scaled_to(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 0.01:
            print(f"[Calibration] WARNING: aspect ratio changed "
                  f"({self.width}x{self.height} -> {width}x{height}); recalibrate.")
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                          self.dist, width, height, self.source + " (rescaled)", self.rms)

    def describe(self):
        return (f"fx={self.fx:.1f}px fy={self.fy:.1f}px cx={self.cx:.1f} cy={self.cy:.1f} "
                f"@ {self.width}x{self.height}, HFOV={self.hfov_deg:.1f} deg, source={self.source}"
                + (f", RMS reproj={self.rms:.3f}px" if self.rms is not None else ""))


def load_intrinsics(width=FRAME_WIDTH, height=FRAME_HEIGHT, path=CALIBRATION_FILE):
    """Load calibration (rescaled to width x height) or fall back to FOCAL_LENGTH."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            c = json.load(f)
        intr = Intrinsics(c["fx"], c["fy"], c["cx"], c["cy"], c.get("dist"),
                          c["width"], c["height"], c.get("method", "file"), c.get("rms"))
        return intr.scaled_to(width, height)

    print(f"[Calibration] WARNING: {path} not found - using nominal f={FOCAL_LENGTH}px "
          f"from {NOMINAL_FOV_SOURCE}, no distortion model. Calibrate before reporting results.")
    f = FOCAL_LENGTH * width / FRAME_WIDTH
    return Intrinsics(f, f, width / 2, height / 2, None, width, height,
                      f"nominal ({NOMINAL_FOV_SOURCE})")


def save_intrinsics(intr, path=CALIBRATION_FILE, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "method": intr.source, "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        "dist": None if intr.dist is None else intr.dist.tolist(),
        "rms": intr.rms, "hfov_deg": intr.hfov_deg,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Bounding boxes are measured in this same width x height frame.",
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Calibration] Saved {path}\n  {intr.describe()}")


# ── Single-shot reference ────────────────────────────────────────────────────
def calibrate(known_width_cm, actual_distance_cm, measured_pixel_width):
    """f = (w_b * D) / W_r  — quick field check, NOT a substitute for checkerboard."""
    focal = (measured_pixel_width * actual_distance_cm) / known_width_cm
    print(f"Single-shot focal length: {focal:.2f} px")
    return focal


# ── Checkerboard ─────────────────────────────────────────────────────────────
def _find_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def _collect_from_camera(pattern, min_views):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    views, size = [], None
    print(f"SPACE = capture view (need >= {min_views}, vary angle and position), "
          f"C = calibrate, Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = _find_corners(gray, pattern)
        disp = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(disp, pattern, corners, True)
        cv2.putText(disp, f"views: {len(views)}  {size[0]}x{size[1]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and corners is not None:
            views.append(corners)
            print(f"  captured view {len(views)}")
        elif k == ord("c") and len(views) >= min_views:
            break
        elif k in (ord("q"), 27):
            cap.release(); cv2.destroyAllWindows()
            sys.exit("Calibration cancelled.")
    cap.release(); cv2.destroyAllWindows()
    return views, size


def _collect_from_folder(folder, pattern):
    views, size = [], None
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        s = (img.shape[1], img.shape[0])
        if size and s != size:
            print(f"  skip {p}: size {s} != {size}")
            continue
        size = s
        corners = _find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
        print(f"  {os.path.basename(p)}: {'ok' if corners is not None else 'no board'}")
        if corners is not None:
            views.append(corners)
    return views, size


def run_checkerboard(cols, rows, square_mm, images=None, min_views=12, out=CALIBRATION_FILE):
    pattern = (cols, rows)   # INNER corners
    views, size = _collect_from_folder(images, pattern) if images else _collect_from_camera(pattern, min_views)
    if len(views) < min_views:
        sys.exit(f"Only {len(views)} usable views; need >= {min_views}.")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(views), views, size, None, None)

    intr = Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist, size[0], size[1],
                      "checkerboard calibrateCamera", float(rms))
    save_intrinsics(intr, out, {"views": len(views), "board_inner_corners": [cols, rows],
                                "square_mm": square_mm})
    if (size[0], size[1]) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(f"  NOTE: calibrated at {size}, runtime is {FRAME_WIDTH}x{FRAME_HEIGHT}; "
              f"intrinsics will be rescaled at load time.")
    return intr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cb = sub.add_parser("checkerboard", help="full intrinsic calibration")
    cb.add_argument("--cols", type=int, default=9, help="inner corners per row")
    cb.add_argument("--rows", type=int, default=6, help="inner corners per column")
    cb.add_argument("--square-mm", type=float, required=True)
    cb.add_argument("--images", help="folder of checkerboard images (else live camera)")
    cb.add_argument("--min-views", type=int, default=12)
    cb.add_argument("--out", default=CALIBRATION_FILE)

    ref = sub.add_parser("reference", help="single-shot focal check")
    ref.add_argument("--width-cm", type=float, required=True)
    ref.add_argument("--distance-cm", type=float, required=True)
    ref.add_argument("--pixels", type=float, required=True, help="box width in capture-frame pixels")

    sub.add_parser("show", help="print the intrinsics the system will use")

    a = ap.parse_args(argv)
    if a.cmd == "checkerboard":
        run_checkerboard(a.cols, a.rows, a.square_mm, a.images, a.min_views, a.out)
    elif a.cmd == "reference":
        f = calibrate(a.width_cm, a.distance_cm, a.pixels)
        cur = load_intrinsics()
        print(f"Current fx = {cur.fx:.1f}px  (difference {100 * (f - cur.fx) / cur.fx:+.1f}%)")
    else:
        print(load_intrinsics().describe())


if __name__ == "__main__":
    main()
