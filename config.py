import os
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FPS_TARGET     = 30

MODEL_PATH         = "models/yolov8n.pt"
CONFIDENCE         = 0.35
IOU_THRESHOLD      = 0.45
MAX_DETECTIONS     = 50
DEVICE             = "cpu"
IMGSZ              = 640

FOCAL_LENGTH       = 615   

DISTANCE_DANGER    = 1.0  
DISTANCE_MODERATE  = 2.5  
DISTANCE_GOOD      = 4.0  

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
    "bicycle":       60,
    "car":          180,
    "motorcycle":    80,
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
    "bottle":         7,
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

HIGH_PRIORITY = {
    "person", "car", "truck", "bus",
    "motorcycle", "bicycle", "dog", "cat",
    "horse", "cow", "bear",
    "chair", "couch", "dining table", "bed", "toilet",
    "bench", "suitcase", "backpack",
    "bottle", "cup", "knife", "scissors",
    "stairs", "traffic light", "stop sign", "fire hydrant",
    "potted plant", "umbrella", "sports ball",
    "laptop", "cell phone", "tv", "refrigerator",
}

# ── Voice ────────────────────────────────────
VOICE_RATE     = 160
VOICE_VOLUME   = 1.0
VOICE_COOLDOWN = 2.5   

NAV_GUIDANCE_THRESHOLD = 3.0

# ── Display ──────────────────────────────────
SHOW_FPS       = True
SHOW_DISTANCE  = True
SHOW_LABELS    = True
SHOW_SIDEBAR   = True

# ── Logging ──────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FILE   = "logs/assistant.log"
