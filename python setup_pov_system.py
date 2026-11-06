import os

files = {}

# ════════════════════════════════════════════════════════
# config.py
# ════════════════════════════════════════════════════════
files["config.py"] = '''
import os

CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FPS_TARGET     = 30

MODEL_PATH         = "models/yolov8n.pt"
CONFIDENCE         = 0.45
IOU_THRESHOLD      = 0.45
MAX_DETECTIONS     = 30
DEVICE             = "cpu"

FOCAL_LENGTH       = 615

# Distance ranges (metres)
DISTANCE_DANGER    = 1.2   # RED
DISTANCE_MODERATE  = 3.0   # ORANGE
DISTANCE_GOOD      = 5.0   # YELLOW
# > 5.0             GREEN

KNOWN_WIDTHS = {
    "person": 45, "bicycle": 60, "car": 180,
    "motorcycle": 80, "bus": 250, "truck": 250,
    "cat": 30, "dog": 40, "chair": 50,
    "dining table": 100, "laptop": 35,
    "cell phone": 7, "bottle": 7, "cup": 8,
    "book": 20, "backpack": 35, "handbag": 30,
    "umbrella": 100, "traffic light": 30,
    "fire hydrant": 30, "stop sign": 75,
    "bench": 150, "tv": 120, "keyboard": 40,
    "mouse": 10, "remote": 5, "clock": 30,
    "vase": 15, "scissors": 20, "teddy bear": 25,
    "default": 40,
}

HIGH_PRIORITY = {
    "person", "car", "truck", "bus",
    "motorcycle", "bicycle", "dog", "cat",
    "traffic light", "stop sign", "fire hydrant"
}

VOICE_RATE     = 150
VOICE_VOLUME   = 1.0
VOICE_COOLDOWN = 2.5

SHOW_FPS      = True
SHOW_DISTANCE = True
SHOW_LABELS   = True

LOG_LEVEL = "INFO"
LOG_FILE  = "logs/assistant.log"
'''

# ════════════════════════════════════════════════════════
# camera/webcam.py  — threaded, no freeze
# ════════════════════════════════════════════════════════
files["camera/webcam.py"] = '''
import cv2, threading, time
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera nahi khuli!")
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(0.6)
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()
'''

# ════════════════════════════════════════════════════════
# object_detection/model_loader.py
# ════════════════════════════════════════════════════════
files["object_detection/model_loader.py"] = '''
import os
from ultralytics import YOLO
from config import MODEL_PATH, DEVICE

def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        print("[Model] Downloading yolov8n.pt ...")
        model = YOLO("yolov8n.pt")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
    else:
        model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    print(f"[Model] Loaded on {DEVICE}")
    return model
'''

# ════════════════════════════════════════════════════════
# object_detection/detector.py
# ════════════════════════════════════════════════════════
files["object_detection/detector.py"] = '''
import numpy as np
from object_detection.model_loader import load_model
from config import CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, KNOWN_WIDTHS, HIGH_PRIORITY

class Detector:
    def __init__(self):
        self.model = load_model()

    def detect(self, frame):
        detections = []
        try:
            results = self.model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                return []

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                bw = x2 - x1
                bh = y2 - y1
                if bw < 10 or bh < 10:
                    continue
                cls = names.get(clsids[i], "object")
                detections.append({
                    "class":      cls,
                    "confidence": float(confs[i]),
                    "bbox":       (int(x1), int(y1), int(x2), int(y2)),
                    "bbox_width": float(bw),
                    "bbox_height":float(bh),
                    "known_width":KNOWN_WIDTHS.get(cls, KNOWN_WIDTHS["default"]),
                    "priority":   cls in HIGH_PRIORITY,
                })
        except Exception as e:
            print(f"[Detector] {e}")
        return detections
'''

# ════════════════════════════════════════════════════════
# distance_estimation/estimator.py
# ════════════════════════════════════════════════════════
files["distance_estimation/estimator.py"] = '''
from config import FOCAL_LENGTH, DISTANCE_DANGER, DISTANCE_MODERATE, DISTANCE_GOOD

class Estimator:
    def __init__(self):
        self.focal = FOCAL_LENGTH

    def estimate(self, detections, frame_width):
        for d in detections:
            dist = self._dist(d["known_width"], d["bbox_width"])
            d["distance"]     = dist
            d["danger_level"] = self._level(dist)
            d["position"]     = self._pos(d["bbox"], frame_width)
        return detections

    def _dist(self, real_w, bbox_w):
        if bbox_w <= 0:
            return 99.0
        return round((real_w * self.focal) / (bbox_w * 100), 2)

    def _level(self, d):
        if d <= DISTANCE_DANGER:   return "DANGER"
        if d <= DISTANCE_MODERATE: return "MODERATE"
        if d <= DISTANCE_GOOD:     return "CAUTION"
        return "SAFE"

    def _pos(self, bbox, fw):
        cx = (bbox[0] + bbox[2]) / 2
        t  = fw / 3
        if cx < t:       return "LEFT"
        elif cx < t * 2: return "CENTER"
        else:            return "RIGHT"
'''

# ════════════════════════════════════════════════════════
# utils/draw.py  — POV style clean UI
# ════════════════════════════════════════════════════════
files["utils/draw.py"] = '''
import cv2

COLORS = {
    "DANGER":   (0,   0,   220),
    "MODERATE": (0,   140, 255),
    "CAUTION":  (0,   210, 210),
    "SAFE":     (0,   200, 80),
}

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        level  = d.get("danger_level", "SAFE")
        color  = COLORS.get(level, (200, 200, 200))
        dist   = d.get("distance", 0)
        cls    = d.get("class", "object")
        pos    = d.get("position", "")
        conf   = d.get("confidence", 0)

        # Thick box
        thickness = 3 if d.get("priority") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Corner accents (POV style)
        length = 18
        cv2.line(frame, (x1, y1), (x1+length, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1+length), color, 3)
        cv2.line(frame, (x2, y1), (x2-length, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1+length), color, 3)
        cv2.line(frame, (x1, y2), (x1+length, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2-length), color, 3)
        cv2.line(frame, (x2, y2), (x2-length, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2-length), color, 3)

        # Label
        label = f"{cls}  {dist:.1f}m  {pos}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        lx = max(x1, 0)
        ly = max(y1 - th - 10, 0)
        fh, fw = frame.shape[:2]
        lx = min(lx, fw - tw - 8)

        cv2.rectangle(frame, (lx-2, ly-2), (lx+tw+6, ly+th+6), (0,0,0), -1)
        cv2.rectangle(frame, (lx-2, ly-2), (lx+tw+6, ly+th+6), color, 1)
        cv2.putText(frame, label, (lx+2, ly+th),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

        # Danger badge
        if level == "DANGER":
            badge = "!! DANGER !!"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            bx = x1 + (x2-x1)//2 - bw//2
            by = y2 + bh + 6
            if by < fh - 5:
                cv2.putText(frame, badge, (bx, by),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65,
                            (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps, detections):
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, f"FPS {fps:.0f}",
                (12, 29), cv2.FONT_HERSHEY_DUPLEX,
                0.70, (0, 255, 160), 1, cv2.LINE_AA)

    count = len(detections)
    cv2.putText(frame, f"Objects: {count}",
                (130, 29), cv2.FONT_HERSHEY_DUPLEX,
                0.70, (0, 255, 160), 1, cv2.LINE_AA)

    # Closest object info in top bar
    if detections:
        priority_order = {"DANGER":0,"MODERATE":1,"CAUTION":2,"SAFE":3}
        best = min(detections,
                   key=lambda d: priority_order.get(d.get("danger_level","SAFE"), 3))
        level = best.get("danger_level","SAFE")
        cls   = best.get("class","")
        dist  = best.get("distance",0)
        pos   = best.get("position","")
        info  = f"Nearest: {cls}  {dist:.1f}m  {pos}  [{level}]"
        color = {
            "DANGER":(0,0,220),"MODERATE":(0,140,255),
            "CAUTION":(0,210,210),"SAFE":(0,200,80)
        }.get(level,(200,200,200))
        cv2.putText(frame, info,
                    (w//2 - 200, 29),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.60, color, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q=Quit  V=Voice  S=Screenshot",
                (w - 340, 29),
                cv2.FONT_HERSHEY_DUPLEX,
                0.50, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Bottom legend bar ────────────────────────
    legend = [
        ("DANGER <1.2m",   (0,   0,   220)),
        ("MODERATE 1.2-3m",(0,   140, 255)),
        ("CAUTION 3-5m",   (0,   210, 210)),
        ("SAFE >5m",       (0,   200, 80)),
    ]
    bar_h = 28
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-bar_h), (w, h), (15,15,15), -1)
    cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)

    x_leg = 10
    for txt, col in legend:
        cv2.circle(frame, (x_leg+7, h-bar_h//2), 6, col, -1)
        cv2.putText(frame, txt,
                    (x_leg+18, h-bar_h//2+5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.45, col, 1, cv2.LINE_AA)
        (tw,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        x_leg += tw + 40

    # ── Crosshair center ─────────────────────────
    cx, cy = w//2, h//2
    cv2.line(frame, (cx-15, cy), (cx+15, cy), (255,255,255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy-15), (cx, cy+15), (255,255,255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx,cy), 4, (255,255,255), 1, cv2.LINE_AA)

    return frame
'''

# ════════════════════════════════════════════════════════
# utils/fps.py
# ════════════════════════════════════════════════════════
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
        return (len(self.times)-1) / (self.times[-1]-self.times[0])
'''

# ════════════════════════════════════════════════════════
# utils/logger.py
# ════════════════════════════════════════════════════════
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
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
'''

# ════════════════════════════════════════════════════════
# voice/speaker.py
# ════════════════════════════════════════════════════════
files["voice/speaker.py"] = '''
import pyttsx3, threading, time, queue
from config import VOICE_RATE, VOICE_VOLUME, VOICE_COOLDOWN

class Speaker:
    def __init__(self):
        self.q         = queue.Queue(maxsize=1)
        self.last_time = 0
        self.enabled   = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.q.get()
            if text and self.enabled:
                try:
                    eng = pyttsx3.init()
                    eng.setProperty("rate",   VOICE_RATE)
                    eng.setProperty("volume", VOICE_VOLUME)
                    eng.say(text)
                    eng.runAndWait()
                    eng.stop()
                except Exception as e:
                    print(f"[Speaker] {e}")
            self.q.task_done()

    def speak(self, detections):
        if not self.enabled:
            return
        if time.time() - self.last_time < VOICE_COOLDOWN:
            return
        if not detections:
            return

        order = {"DANGER":0,"MODERATE":1,"CAUTION":2,"SAFE":3}
        best  = min(detections,
                    key=lambda d: order.get(d.get("danger_level","SAFE"), 3))

        level = best.get("danger_level","SAFE")
        cls   = best.get("class","object")
        dist  = best.get("distance", 0)
        pos   = best.get("position","ahead")

        if level == "DANGER":
            msg = f"Danger! {cls} very close. {dist:.1f} metres on your {pos.lower()}. Stop!"
        elif level == "MODERATE":
            msg = f"Warning. {cls} at {dist:.1f} metres on your {pos.lower()}."
        elif level == "CAUTION":
            msg = f"{cls} detected. {dist:.1f} metres, {pos.lower()}."
        else:
            return

        self.last_time = time.time()
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            pass

    def toggle(self):
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"[Speaker] Voice {state}")
        try:
            self.q.put_nowait(f"Voice {state}")
        except queue.Full:
            pass
'''

# ════════════════════════════════════════════════════════
# main.py  — POV walking style
# ════════════════════════════════════════════════════════
files["main.py"] = '''
import cv2, time, os
from camera.webcam                  import Webcam
from object_detection.detector      import Detector
from distance_estimation.estimator  import Estimator
from voice.speaker                  import Speaker
from utils.draw                     import draw_detections, draw_hud
from utils.fps                      import FPSCounter

def main():
    print("=" * 45)
    print("   Blind Assistance System — POV Mode")
    print("=" * 45)

    cam = Webcam().start()
    det = Detector()
    est = Estimator()
    spk = Speaker()
    fps = FPSCounter()

    os.makedirs("screenshots", exist_ok=True)

    DETECT_EVERY = 0.12        # detect every 120ms
    last_detect  = 0
    detections   = []          # always fresh

    print("[READY]  Q=Quit  V=Voice  S=Screenshot")
    spk.speak([{
        "class":"system","danger_level":"CAUTION",
        "distance":1.0,"position":"CENTER"
    }])

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        fps.update()
        now = time.time()

        # ── Detect on interval ────────────────────
        if now - last_detect >= DETECT_EVERY:
            last_detect = now
            raw = det.detect(frame)

            # KEY: empty camera = empty detections
            if len(raw) == 0:
                detections = []
            else:
                detections = est.estimate(raw, frame.shape[1])
                spk.speak(detections)

        # ── Render ───────────────────────────────
        display = frame.copy()
        display = draw_detections(display, detections)
        display = draw_hud(display, fps.get(), detections)

        cv2.imshow("Blind Assistance  [POV]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("v"):
            spk.toggle()
        elif key == ord("s"):
            fname = f"screenshots/snap_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"[Screenshot] {fname}")

    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

if __name__ == "__main__":
    main()
'''

# ════════════════════════════════════════════════════════
# __init__.py files
# ════════════════════════════════════════════════════════
files["camera/__init__.py"]              = "from camera.webcam import Webcam"
files["object_detection/__init__.py"]    = "from object_detection.detector import Detector"
files["distance_estimation/__init__.py"] = "from distance_estimation.estimator import Estimator"
files["voice/__init__.py"]               = "from voice.speaker import Speaker"
files["tracking/__init__.py"]            = ""
files["navigation/__init__.py"]          = ""
files["utils/__init__.py"]               = ""

# ════════════════════════════════════════════════════════
# WRITE FILES
# ════════════════════════════════════════════════════════
print("Writing all files...")
for path, content in files.items():
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"  OK  {path}")

os.makedirs("models",      exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ════════════════════════════════════════════════════════
# TEST IMPORTS
# ════════════════════════════════════════════════════════
print("\nTesting imports...")
tests = [
    ("config",                       "CONFIDENCE"),
    ("camera.webcam",                "Webcam"),
    ("object_detection.detector",    "Detector"),
    ("distance_estimation.estimator","Estimator"),
    ("voice.speaker",                "Speaker"),
    ("utils.draw",                   "draw_detections"),
    ("utils.draw",                   "draw_hud"),
    ("utils.fps",                    "FPSCounter"),
    ("utils.logger",                 "get_logger"),
]
ok = True
for mod, attr in tests:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"  OK  {mod}.{attr}")
    except Exception as e:
        print(f"  FAIL  {mod}.{attr}  =>  {e}")
        ok = False

print()
if ok:
    print("ALL OK! Run:  python main.py")
else:
    print("Fix errors above then run: python main.py")
