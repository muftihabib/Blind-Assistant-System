"""
Central configuration for the Blind Assistance System.

Every number that appears in the paper (thresholds, priors, timings) is
defined here so that the code and the paper can be checked against each
other in one place.
"""

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280          # capture resolution; bounding boxes are in THIS frame
FRAME_HEIGHT   = 720
FPS_TARGET     = 30

# ── Detector ─────────────────────────────────────────────────────────────────
MODEL_PATH         = "models/yolov8n.pt"
CONFIDENCE         = 0.35
IOU_THRESHOLD      = 0.45      # NMS IoU
MAX_DETECTIONS     = 50
DEVICE             = "cpu"
IMGSZ              = 640       # letterboxed network input (done inside Ultralytics)
DETECT_EVERY_N     = 1         # run the detector on every Nth frame (1 = every frame)
TORCH_THREADS      = None      # CPU threads for inference (None = physical cores)

# ── Pre-processing (applied to the frame after it is downscaled to IMGSZ wide)
PREPROCESS_DENOISE = True      # 3x3 Gaussian blur, sigma auto
PREPROCESS_CLAHE   = True      # CLAHE on the L channel of CIE-LAB
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = 8         # 8x8 tiles

# ── Camera intrinsics / distance ─────────────────────────────────────────────
# Written by:  python -m distance_estimation.calibration checkerboard ...
CALIBRATION_FILE   = "calibration/camera_intrinsics.json"
# Fallback used ONLY when no calibration file exists: nominal focal length from the
# manufacturer's horizontal field of view (Logitech C920: HFOV 70.42 deg, VFOV 43.3 deg,
# Logitech "C920 Technical Specifications"). f = (W/2) / tan(HFOV/2) = 906.9 px at 1280 px.
# Run the checkerboard calibration before reporting measured results.
import math as _math
NOMINAL_HFOV_DEG   = 70.42
NOMINAL_FOV_SOURCE = "Logitech C920 technical specifications (HFOV 70.42 deg)"
FOCAL_LENGTH       = round((FRAME_WIDTH / 2) / _math.tan(_math.radians(NOMINAL_HFOV_DEG / 2)), 1)

# Danger zones (metres) — Table III
DISTANCE_DANGER    = 1.0       # D <= 1.0          -> DANGER  (red)
DISTANCE_WARNING   = 2.5       # 1.0 < D <= 2.5    -> WARNING (orange)
DISTANCE_CAUTION   = 4.0       # 2.5 < D <= 4.0    -> CAUTION (yellow)
                               # D > 4.0           -> SAFE    (green)

# A box whose left/right edge is within this many pixels of the frame border is
# treated as CLIPPED: its width is truncated, so the pinhole distance is only an
# upper bound on the true distance.
EDGE_MARGIN_PX         = 4
# A clipped box that is at least this fraction of the frame width is escalated
# to DANGER regardless of the (over-)estimated distance.
CLIPPED_DANGER_FRAC    = 0.5

# Occlusion: a box overlapping a foreground box (lower bottom edge) with
# IoU > this value has its distance flagged as unreliable ("estimated").
OCCLUSION_IOU          = 0.40

# Temporal smoothing of distance per track (moving average, frames) — §III-E.3
SMOOTHING_WINDOW       = 5

# Frontal real-world widths W_r,front (cm) — Table IV and extended COCO priors
KNOWN_WIDTHS = {
    "person":        45,
    "cat":           30,
    "dog":           40,
    "horse":        200,
    "sheep":         60,
    "cow":          180,
    "elephant":     350,
    "bear":         120,
    "zebra":        200,
    "giraffe":      150,
    "bird":          20,
    # ── Vehicles ──
    "bicycle":       70,       # handlebars (Table IV)
    "car":          180,       # bumper     (Table IV)
    "motorcycle":    70,       # handlebars (Table IV)
    "bus":          250,
    "truck":        250,
    "airplane":     3000,
    "boat":         300,
    "train":        300,
    # ── Street objects ──
    "traffic light": 30,
    "fire hydrant":  30,
    "stop sign":     75,
    "parking meter": 20,
    "bench":        150,
    # ── Furniture ──
    "chair":         50,
    "couch":        180,
    "bed":          150,
    "dining table": 100,
    "toilet":        40,
    # ── Electronics ──
    "tv":           120,
    "laptop":        35,
    "cell phone":     7,
    "keyboard":      40,
    "mouse":         10,
    "remote":         5,
    "microwave":     50,
    "oven":          60,
    "toaster":       30,
    "refrigerator":  70,
    # ── Kitchen / food ──
    "bottle":         8,       # Table IV
    "cup":            8,
    "wine glass":     7,
    "fork":          15,
    "knife":         15,
    "spoon":         12,
    "bowl":          18,
    "banana":        18,
    "apple":          8,
    "sandwich":      15,
    "orange":         8,
    "broccoli":      15,
    "carrot":        15,
    "hot dog":       15,
    "pizza":         30,
    "donut":         10,
    "cake":          25,
    # ── Accessories ──
    "umbrella":      100,
    "backpack":      35,
    "handbag":       30,
    "suitcase":      50,
    "tie":           10,
    # ── Sports ──
    "frisbee":       25,
    "skis":          15,
    "snowboard":     30,
    "sports ball":   22,
    "kite":          80,
    "baseball bat":  70,
    "baseball glove": 25,
    "skateboard":    20,
    "surfboard":     50,
    "tennis racket": 28,
    # ── Misc objects ──
    "book":          20,
    "clock":         30,
    "vase":          15,
    "scissors":      20,
    "teddy bear":    25,
    "hair drier":    20,
    "toothbrush":     2,
    "potted plant":  30,
    "sink":          50,
    "default":       40,
}

# Side-profile widths W_r,side (cm) and the aspect-ratio trigger (w_b / h_b) above
# which the side width is used — Table IV. Only classes listed here switch.
# (Person has no entry: a person's box aspect ratio does not separate frontal
#  from side views, so no trigger can select a side width for it.)
SIDE_PROFILE = {
    #  class         (W_r,side cm, AR trigger)
    "bicycle":     (175, 1.2),
    "motorcycle":  (175, 1.2),
    "car":         (420, 1.4),
}

# ── Tracking ─────────────────────────────────────────────────────────────────
TRACK_MAX_MISSING      = 5     # frames a track survives without a match
TRACK_MAX_MATCH_FRAC   = 0.10  # max centroid jump, as a fraction of frame width

# ── Path guidance (§III-G) ───────────────────────────────────────────────────
SECTOR_SPLITS          = (1 / 3, 2 / 3)   # LEFT | CENTER | RIGHT boundaries
GUIDANCE_TRIGGER       = 2.5   # centre obstacle within this distance -> steer
CLEAR_DISTANCE         = 3.0   # a side sector is Clear iff every obstacle in it is farther
SECTOR_MIN_OVERLAP     = 0.10  # box must cover >=10% of a sector's width to occupy it

# ── Voice alerts (§III-H) ────────────────────────────────────────────────────
TTS_BACKEND            = "pyttsx3"   # "pyttsx3" (offline, default) or "sapi"
VOICE_RATE             = 190         # words per minute (pyttsx3); alerts must be short
VOICE_VOLUME           = 1.0
ALERT_COOLDOWN         = 3.0   # seconds per tracked object
APPROACH_DELTA         = 0.5   # metres closer since last alert -> override cooldown
ANNOUNCE_LEVELS        = ("DANGER", "WARNING", "CAUTION")   # SAFE is not spoken

# Classes ranked above others at equal danger level (people and vehicles)
HIGH_PRIORITY = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "dog", "horse", "cow", "bear",
}

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR            = "logs"
ALERT_LOG_FILE     = "logs/alerts.csv"   # one row per spoken alert, with timings

# ── Display ──────────────────────────────────────────────────────────────────
SHOW_SECTORS   = True
