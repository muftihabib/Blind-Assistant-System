"""
Frame pre-processing applied before detection (§III-B).

1. Denoise : 3x3 Gaussian blur (sigma derived from kernel size by OpenCV,
             sigma = 0.3*((3-1)*0.5 - 1) + 0.8 = 0.8).
2. Contrast: CLAHE on the L channel of CIE-LAB (clip limit CLAHE_CLIP_LIMIT,
             CLAHE_TILE_GRID x CLAHE_TILE_GRID tiles); chroma is untouched.

Order of operations in Detector.predict_raw():
    capture frame (e.g. 1280x720)
      -> downscale to IMGSZ wide, aspect kept (e.g. 640x360)   [INTER_AREA]
      -> denoise + CLAHE (this module)
      -> Ultralytics letterbox to IMGSZ x IMGSZ, normalise to [0, 1], YOLOv8n
      -> boxes scaled back by (capture width / IMGSZ) to CAPTURE-frame pixels.
Downscaling first does not change what the network sees (it letterboxes to
IMGSZ anyway) but makes pre-processing ~4x cheaper. All distances use
capture-frame box widths with capture-frame intrinsics.
"""

import cv2

from config import (CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, PREPROCESS_CLAHE,
                    PREPROCESS_DENOISE)


class Preprocessor:
    def __init__(self, denoise=PREPROCESS_DENOISE, clahe=PREPROCESS_CLAHE):
        self.denoise = denoise
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                     tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID)) if clahe else None

    def __call__(self, frame):
        out = frame
        if self.denoise:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        if self.clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            out = cv2.cvtColor(cv2.merge((self.clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        return out
