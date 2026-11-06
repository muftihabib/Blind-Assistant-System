import os, sys

files = {}

# ─────────────────────────────────────────────
# config.py  –  sahi distance ranges ke saath
# ─────────────────────────────────────────────
files["config.py"] = '''
import os

# ── Camera ──────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FPS_TARGET     = 30

# ── Model ───────────────────────────────────
MODEL_PATH         = "models/yolov8n.pt"
CONFIDENCE         = 0.40
IOU_THRESHOLD      = 0.45
MAX_DETECTIONS     = 50
DEVICE             = "cpu"

# ── Distance (meters) ────────────────────────
FOCAL_LENGTH       = 615   # pixels (calibrated)

DISTANCE_DANGER    = 1.2   # RED   – bohot paas
DISTANCE_MODERATE  = 3.0   # ORANGE – moderate
DISTANCE_GOOD      = 5.0   # YELLOW – thoda door
# > 5.0 m           → GREEN  – safe

# ── Known real-world widths (cm) ─────────────
KNOWN_WIDTHS = {
    "person":        45,
    "bicycle":       60,
    "car":          180,
    "motorcycle":    80,
    "bus":          250,
    "truck":        250,
    "cat":           30,
    "dog":           40,
    "chair":         50,
    "dining table": 100,
    "laptop":        35,
    "cell phone":     7,
    "bottle":         7,
    "cup":            8,
    "book":          20,
    "backpack":      35,
    "handbag":       30,
    "umbrella":      100,
    "traffic light": 30,
    "fire hydrant":  30,
    "stop sign":     75,
    "bench":        150,
    "tv":           120,
    "keyboard":      40,
    "mouse":         10,
    "remote":         5,
    "clock":         30,
    "vase":          15,
    "scissors":      20,
    "teddy bear":    25,
    "default":       40,
}

# ── High-priority objects ────────────────────
HIGH_PRIORITY = {
    "person","car","truck","bus",
    "motorcycle","bicycle","dog","cat",
    "stairs","traffic light","stop sign","fire hydrant"
}

# ── Voice ────────────────────────────────────
VOICE_RATE     = 155
VOICE_VOLUME   = 1.0
VOICE_COOLDOWN = 3.0   # seconds between alerts

# ── Display ──────────────────────────────────
SHOW_FPS       = True
SHOW_DISTANCE  = True
SHOW_LABELS    = True
SHOW_SIDEBAR   = True

# ── Logging ──────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FILE   = "logs/assistant.log"
'''

# ─────────────────────────────────────────────
# camera/webcam.py  –  separate thread so frame
# never sticks / freezes
# ─────────────────────────────────────────────
files["camera/webcam.py"] = '''
import cv2, threading, time
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class Webcam:
    def __init__(self):
        self.cap    = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ← key fix: 1-frame buffer
        self.frame  = None
        self.lock   = threading.Lock()
        self.running = False

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera nahi khuli! Index check karo.")
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(0.5)   # warm-up
        return self

    def _reader(self):
        """Background thread – hamesha latest frame rakhta hai"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        time.sleep(0.3)
        self.cap.release()
'''

# ─────────────────────────────────────────────
# object_detection/detector.py
# ─────────────────────────────────────────────
files["object_detection/detector.py"] = '''
import cv2
import numpy as np
from object_detection.model_loader import load_model
from config import CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, KNOWN_WIDTHS, HIGH_PRIORITY

class Detector:
    def __init__(self):
        self.model = load_model()

    def detect(self, frame):
        """Return list of detection dicts for ALL objects in frame"""
        detections = []
        try:
            results = self.model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                verbose=False,
                stream=False,
            )
            if not results:
                return detections

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            # ── bulk numpy extract ──────────────────
            boxes_xyxy  = result.boxes.xyxy.cpu().numpy()   # [N,4]
            confs       = result.boxes.conf.cpu().numpy()    # [N]
            cls_ids     = result.boxes.cls.cpu().numpy().astype(int)  # [N]
            names       = result.names                       # dict

            for i in range(len(boxes_xyxy)):
                x1,y1,x2,y2 = boxes_xyxy[i]
                conf         = float(confs[i])
                cls_id       = cls_ids[i]
                class_name   = names.get(cls_id, "object")

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w < 5 or bbox_h < 5:   # too small – skip noise
                    continue

                detections.append({
                    "class":       class_name,
                    "confidence":  conf,
                    "bbox":        (int(x1), int(y1), int(x2), int(y2)),
                    "bbox_width":  float(bbox_w),
                    "bbox_height": float(bbox_h),
                    "known_width": KNOWN_WIDTHS.get(class_name,
                                   KNOWN_WIDTHS["default"]),
                    "priority":    class_name in HIGH_PRIORITY,
                })

        except Exception as e:
            print(f"[Detector] Error: {e}")

        return detections
'''

# ─────────────────────────────────────────────
# distance_estimation/estimator.py
# ─────────────────────────────────────────────
files["distance_estimation/estimator.py"] = '''
from config import (FOCAL_LENGTH, KNOWN_WIDTHS,
                    DISTANCE_DANGER, DISTANCE_MODERATE, DISTANCE_GOOD)

class Estimator:
    def __init__(self):
        self.focal = FOCAL_LENGTH

    def estimate(self, detections, frame_width):
        """Add distance + danger_level + position to each detection"""
        for d in detections:
            dist = self._calc_distance(d["known_width"], d["bbox_width"])
            d["distance"]     = dist
            d["danger_level"] = self._danger_level(dist)
            d["position"]     = self._position(d["bbox"], frame_width)
        return detections

    def _calc_distance(self, real_width_cm, bbox_width_px):
        if bbox_width_px <= 0:
            return 999.0
        dist_cm = (real_width_cm * self.focal) / bbox_width_px
        return round(dist_cm / 100, 2)   # convert to metres

    def _danger_level(self, dist_m):
        """
        RED    (DANGER)   → 0 – 1.2 m
        ORANGE (MODERATE) → 1.2 – 3.0 m
        YELLOW (CAUTION)  → 3.0 – 5.0 m
        GREEN  (SAFE)     → > 5.0 m
        """
        if dist_m <= DISTANCE_DANGER:
            return "DANGER"
        elif dist_m <= DISTANCE_MODERATE:
            return "MODERATE"
        elif dist_m <= DISTANCE_GOOD:
            return "CAUTION"
        else:
            return "SAFE"

    def _position(self, bbox, frame_width):
        cx = (bbox[0] + bbox[2]) / 2
        third = frame_width / 3
        if cx < third:
            return "LEFT"
        elif cx < third * 2:
            return "CENTER"
        else:
            return "RIGHT"
'''

# ─────────────────────────────────────────────
# utils/draw.py  –  4-colour system
# ─────────────────────────────────────────────
files["utils/draw.py"] = '''
import cv2

LEVEL_COLORS = {
    "DANGER":   (0,   0,   220),   # RED
    "MODERATE": (0,   140, 255),   # ORANGE
    "CAUTION":  (0,   220, 220),   # YELLOW
    "SAFE":     (0,   200, 80),    # GREEN
}

LEVEL_LABEL = {
    "DANGER":   "DANGER",
    "MODERATE": "MODERATE",
    "CAUTION":  "CAUTION",
    "SAFE":     "SAFE",
}

def draw_detections(frame, detections):
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        level  = d.get("danger_level", "SAFE")
        color  = LEVEL_COLORS.get(level, (200,200,200))
        dist   = d.get("distance", 0)
        label  = (f\'{d["class"]} | {LEVEL_LABEL[level]} | {dist:.1f}m\')

        # bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = max(x1, 0)
        ly = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (lx, ly), (lx+tw+6, ly+th+6), color, -1)
        cv2.putText(frame, label, (lx+3, ly+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return frame

def draw_hud(frame, fps, count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 36), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,180), 2)
    cv2.putText(frame, f"Objects: {count}", (160, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,180), 2)

    # legend bottom-left
    legend = [
        ("DANGER   < 1.2m",  (0,0,220)),
        ("MODERATE 1.2-3m",  (0,140,255)),
        ("CAUTION  3-5m",    (0,220,220)),
        ("SAFE     > 5m",    (0,200,80)),
    ]
    y0 = h - 10
    for txt, col in reversed(legend):
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame,(5,y0-th-4),(tw+12,y0+2),(30,30,30),-1)
        cv2.putText(frame, txt, (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y0 -= (th + 10)
    return frame
'''

# ─────────────────────────────────────────────
# voice/speaker.py
# ─────────────────────────────────────────────
files["voice/speaker.py"] = '''
import pyttsx3, threading, time, queue, pythoncom
from config import VOICE_RATE, VOICE_VOLUME, VOICE_COOLDOWN

class Speaker:
    def __init__(self):
        self.q          = queue.Queue(maxsize=1)
        self.object_states = {}
        self.next_object_id = 0
        self.enabled    = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        pythoncom.CoInitialize()
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", VOICE_RATE)
            engine.setProperty("volume", VOICE_VOLUME)
            while True:
                text = self.q.get()
                try:
                    if text and self.enabled:
                        engine.say(text)
                        engine.runAndWait()
                except Exception as e:
                    print(f"[Speaker] {e}")
                finally:
                    self.q.task_done()
        finally:
            pythoncom.CoUninitialize()

    def clear(self):
        self.object_states.clear()
        self.next_object_id = 0
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break

    def speak(self, detections):
        """Speak every newly detected object in danger-first order."""
        if not self.enabled:
            return
        if not detections:
            return

        ordered = sorted(detections, key=lambda d: d.get("distance", 999.0))
        now = time.time()
        messages = []
        pending_ids = []
        current_states = {}

        for detection in ordered:
            level = detection.get("danger_level", "SAFE")
            cls = detection.get("class", "object")
            dist = detection.get("distance", 0)
            pos = detection.get("position", "ahead")
            bbox = detection.get("bbox", (0, 0, 0, 0))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            matched_id = None
            nearest_distance = float("inf")
            for object_id, state in self.object_states.items():
                if state["class"] != cls or object_id in current_states:
                    continue
                movement = ((center[0] - state["center"][0]) ** 2 +
                            (center[1] - state["center"][1]) ** 2) ** 0.5
                if movement < nearest_distance and movement < 180:
                    nearest_distance = movement
                    matched_id = object_id

            if matched_id is None:
                matched_id = self.next_object_id
                self.next_object_id += 1
                moved = True
                last_spoken = 0
            else:
                moved = nearest_distance >= 50
                last_spoken = self.object_states[matched_id]["last_spoken"]

            current_states[matched_id] = {
                "class": cls,
                "center": center,
                "last_spoken": last_spoken,
            }

            if not moved and now - last_spoken < VOICE_COOLDOWN:
                continue

            if level == "DANGER":
                message = f"Danger! {cls} very close, {dist:.1f} metres, on your {pos}"
            elif level == "MODERATE":
                message = f"Warning. {cls} approaching, {dist:.1f} metres {pos}"
            elif level == "CAUTION":
                message = f"Caution. {cls} detected, {dist:.1f} metres {pos}"
            else:
                message = f"{cls} detected, {dist:.1f} metres {pos}"

            messages.append(message)
            pending_ids.append(matched_id)

        if not messages:
            return

        text = ". ".join(messages)
        try:
            self.q.put_nowait(text)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.task_done()
                self.q.put_nowait(text)
            except queue.Empty:
                return
            except queue.Full:
                return

        for object_id in pending_ids:
            current_states[object_id]["last_spoken"] = now
        self.object_states = current_states

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[Speaker] Voice {'ON' if self.enabled else 'OFF'}")
'''

# ─────────────────────────────────────────────
# main.py  –  threaded camera, 4-level colours
# ─────────────────────────────────────────────
files["main.py"] = '''
import cv2, time, sys
from camera.webcam                  import Webcam
from object_detection.detector      import Detector
from distance_estimation.estimator  import Estimator
from voice.speaker                  import Speaker
from utils.draw                     import draw_detections, draw_hud
from utils.fps                      import FPSCounter

def main():
    print("=== Blind Assistance System Starting ===")

    # ── Init ─────────────────────────────────
    cam  = Webcam().start()
    det  = Detector()
    est  = Estimator()
    spk  = Speaker()
    fps  = FPSCounter()

    print("[INFO] System ready. Press Q=Quit  V=Voice toggle")

    SKIP = 2          # process every 2nd frame to avoid freeze
    fc   = 0
    last_detections = []

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        fc += 1
        fps.update()

        # ── Detection every SKIP frames ───────
        if fc % SKIP == 0:
            detections      = det.detect(frame)
            last_detections = est.estimate(detections, frame.shape[1])
            spk.speak(last_detections)

        # ── Draw ─────────────────────────────
        display = frame.copy()
        display = draw_detections(display, last_detections)
        display = draw_hud(display, fps.get(), len(last_detections))

        cv2.imshow("Blind Assistance System  |  Q=Quit  V=Voice", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("v"):
            spk.toggle()

    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

if __name__ == "__main__":
    main()
'''

# ─────────────────────────────────────────────
# utils/fps.py
# ─────────────────────────────────────────────
files["utils/fps.py"] = '''
import time

class FPSCounter:
    def __init__(self, avg=30):
        self.avg   = avg
        self.times = []

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.avg:
            self.times.pop(0)

    def get(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times)-1) / (self.times[-1] - self.times[0])
'''

# ─────────────────────────────────────────────
# utils/logger.py
# ─────────────────────────────────────────────
files["utils/logger.py"] = '''
import logging, os
from config import LOG_LEVEL, LOG_FILE

def get_logger(name="assistant"):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger
'''

# ─────────────────────────────────────────────
# object_detection/model_loader.py
# ─────────────────────────────────────────────
files["object_detection/model_loader.py"] = '''
import os
from ultralytics import YOLO
from config import MODEL_PATH, DEVICE

def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        print(f"[ModelLoader] {MODEL_PATH} missing/corrupt – downloading yolov8n.pt …")
        model = YOLO("yolov8n.pt")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
    else:
        model = YOLO(MODEL_PATH)

    model.to(DEVICE)
    print(f"[ModelLoader] Model loaded on: {DEVICE}")
    return model
'''

# ─────────────────────────────────────────────
# __init__.py for every sub-package
# ─────────────────────────────────────────────
init_map = {
    "camera/__init__.py":              "from camera.webcam import Webcam",
    "object_detection/__init__.py":    "from object_detection.detector import Detector",
    "distance_estimation/__init__.py": "from distance_estimation.estimator import Estimator",
    "voice/__init__.py":               "from voice.speaker import Speaker",
    "tracking/__init__.py":            "",
    "navigation/__init__.py":          "",
    "utils/__init__.py":               "",
}
files.update(init_map)

# ═════════════════════════════════════════════
# WRITE ALL FILES
# ═════════════════════════════════════════════
print("Writing files …")
for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"  ✓  {path}")

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ═════════════════════════════════════════════
# IMPORT TESTS
# ═════════════════════════════════════════════
print("\nTesting imports …")
tests = [
    ("config",                      "CONFIDENCE"),
    ("camera.webcam",               "Webcam"),
    ("object_detection.detector",   "Detector"),
    ("distance_estimation.estimator","Estimator"),
    ("voice.speaker",               "Speaker"),
    ("utils.draw",                  "draw_detections"),
    ("utils.fps",                   "FPSCounter"),
    ("utils.logger",                "get_logger"),
]
all_ok = True
for mod, attr in tests:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"  ✓  {mod}")
    except Exception as e:
        print(f"  ✗  {mod}  →  {e}")
        all_ok = False

if all_ok:
    print("\n  ALL OK!  Ab chalaao:  python main.py")
else:
    print("\n  Kuch imports fail hue – upar error dekho")
