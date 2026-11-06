import os
print("Writing all fixed detection files...\n")

# ================================================================
# 1. config.py - optimized settings
config_content = """
CAMERA_INDEX    = 0
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FPS_TARGET      = 30

MODEL_PATH      = "models/yolov8n.pt"
YOLO_FALLBACK   = "yolov8n.pt"
CONFIDENCE      = 0.35
IOU_THRESHOLD   = 0.45
MAX_DETECTIONS  = 100
DEVICE          = "cpu"

FOCAL_LENGTH    = 615
DANGER_DIST     = 1.0
WARNING_DIST    = 2.5
CAUTION_DIST    = 4.0

VOICE_RATE      = 155
VOICE_VOLUME    = 1.0
VOICE_COOLDOWN  = 3
VOICE_ENABLED   = True

SHOW_FPS        = True
SHOW_SIDEBAR    = True
SHOW_DISTANCE   = True
WINDOW_TITLE    = "Vision Assistant - Blind Assistance System"

LOG_LEVEL       = "INFO"
LOG_FILE        = "vision_assistant.log"
"""

with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_content.lstrip())
print("  Written: config.py")

# ================================================================
# 2. object_detection/inference.py - fixed for multiple objects
# ================================================================
inference_content = """
from config import CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS

def run_inference(model, frame):
    \"\"\"
    Run YOLO inference.
    Returns results list — always processes ALL detected objects.
    \"\"\"
    try:
        results = model.predict(
            source    = frame,
            conf      = CONFIDENCE,
            iou       = IOU_THRESHOLD,
            max_det   = MAX_DETECTIONS,
            verbose   = False,
            stream    = False,
            agnostic_nms = False,
        )
        return results
    except Exception as e:
        print(f"Inference error: {e}")
        return []
"""

with open("object_detection/inference.py", "w", encoding="utf-8") as f:
    f.write(inference_content.lstrip())
print("  Written: object_detection/inference.py")

# ================================================================
# 3. object_detection/detector.py - fixed multi-object loop
# ================================================================
detector_content = """
from object_detection.model_loader import load_model
from object_detection.inference    import run_inference
from object_detection.classes      import KNOWN_WIDTHS, HIGH_PRIORITY

class Detector:
    def __init__(self):
        self.model = load_model()
        print("Detector ready.")

    def detect(self, frame):
        \"\"\"
        Detect ALL objects in frame.
        Returns list of detection dicts.
        \"\"\"
        detections = []

        try:
            results = run_inference(self.model, frame)
        except Exception as e:
            print(f"Detection error: {e}")
            return []

        for result in results:
            # Safety check
            if result is None or result.boxes is None:
                continue

            boxes = result.boxes

            # Get total number of detections
            num_boxes = len(boxes)

            if num_boxes == 0:
                continue

            # Get all values at once (faster than per-box)
            try:
                all_cls   = boxes.cls.cpu().numpy()
                all_conf  = boxes.conf.cpu().numpy()
                all_xyxy  = boxes.xyxy.cpu().numpy()
            except Exception as e:
                print(f"Box extraction error: {e}")
                continue

            # Loop through every single detection
            for i in range(num_boxes):
                try:
                    cls_id   = int(all_cls[i])
                    cls_name = self.model.names[cls_id]
                    conf     = float(all_conf[i])
                    x1       = int(all_xyxy[i][0])
                    y1       = int(all_xyxy[i][1])
                    x2       = int(all_xyxy[i][2])
                    y2       = int(all_xyxy[i][3])

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    px_width  = x2 - x1
                    px_height = y2 - y1

                    detections.append({
                        "class":        cls_name,
                        "confidence":   conf,
                        "bbox":         (x1, y1, x2, y2),
                        "pixel_width":  px_width,
                        "pixel_height": px_height,
                        "is_priority":  cls_name in HIGH_PRIORITY,
                        "known_width":  KNOWN_WIDTHS.get(cls_name, None),
                    })

                except Exception as e:
                    print(f"  Skipping box {i}: {e}")
                    continue

        return detections
"""

with open("object_detection/detector.py", "w", encoding="utf-8") as f:
    f.write(detector_content.lstrip())
print("  Written: object_detection/detector.py")

# ================================================================
# 4. utils/draw.py - fixed drawing, no overlap glitches
# ================================================================
draw_content = """
import cv2

DANGER_COLORS = {
    "DANGER":  (0,   0,   255),
    "WARNING": (0,  165,  255),
    "CAUTION": (0,  255,  255),
    "SAFE":    (0,  255,    0),
    "UNKNOWN": (180,180,  180),
}

def draw_detection(frame, det):
    \"\"\"Draw bounding box and label for one detection\"\"\"
    try:
        x1, y1, x2, y2 = det["bbox"]
        level  = det.get("danger_level", "UNKNOWN")
        color  = DANGER_COLORS.get(level, (255, 255, 255))
        dist   = det.get("distance_m")
        pos    = det.get("position", "")
        conf   = det.get("confidence", 0)
        name   = det.get("class", "object").upper()

        # Draw main bounding box
        thickness = 3 if level == "DANGER" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        dist_txt = f"{dist}m" if dist else "?m"
        label    = f"{name} {dist_txt} {conf:.0%}"

        # Label background - make sure it stays inside frame
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(
            label, font, font_scale, font_thick)

        label_y = max(y1 - 5, th + 10)
        bg_x1   = x1
        bg_y1   = label_y - th - 6
        bg_x2   = min(x1 + tw + 8, frame.shape[1])
        bg_y2   = label_y + baseline

        # Draw label background
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 4, label_y - 2),
                    font, font_scale, (0, 0, 0), font_thick,
                    cv2.LINE_AA)

        # Draw danger indicator dot
        if level == "DANGER":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    except Exception as e:
        print(f"Draw error: {e}")

    return frame


def draw_hud(frame, fps, obj_count, voice_on):
    \"\"\"Draw heads-up display on frame\"\"\"
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Object count
    cv2.putText(frame, f"Objects: {obj_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Voice status
    voice_txt   = "VOICE: ON" if voice_on else "VOICE: OFF"
    voice_color = (0, 255, 0) if voice_on else (0, 0, 255)
    cv2.putText(frame, voice_txt,
                (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, voice_color, 2, cv2.LINE_AA)

    # Controls
    ctrl = "Q=Quit | V=Voice | S=Screenshot | SPACE=Pause"
    cv2.putText(frame, ctrl,
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_sidebar(frame, detections):
    \"\"\"Draw sorted object list on right side\"\"\"
    h, w  = frame.shape[:2]
    sw    = 270
    sx    = w - sw

    # Sidebar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "DETECTED OBJECTS",
                (sx + 8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (sx, 35), (w, 35), (50, 50, 50), 1)

    # Sort by distance
    DANGER_COLORS_LOCAL = {
        "DANGER":  (0,   0,   255),
        "WARNING": (0,  165,  255),
        "CAUTION": (0,  255,  255),
        "SAFE":    (0,  255,    0),
        "UNKNOWN": (180,180,  180),
    }

    sorted_dets = sorted(
        detections,
        key=lambda d: d.get("distance_m") or 999
    )

    for i, det in enumerate(sorted_dets[:12]):
        y      = 58 + i * 32
        if y > h - 20:
            break

        level  = det.get("danger_level", "UNKNOWN")
        color  = DANGER_COLORS_LOCAL.get(level, (200, 200, 200))
        name   = det["class"][:14]
        dist   = det.get("distance_m")
        pos    = det.get("position", "")

        dist_txt = f"{dist}m" if dist else "?m"
        line1    = f"{name}"
        line2    = f"  {dist_txt} {pos}"

        cv2.putText(frame, line1,
                    (sx + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, line2,
                    (sx + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Total count at bottom
    cv2.putText(frame,
                f"Total: {len(detections)} objects",
                (sx + 8, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 200, 255), 1, cv2.LINE_AA)

    return frame
"""

with open("utils/draw.py", "w", encoding="utf-8") as f:
    f.write(draw_content.lstrip())
print("  Written: utils/draw.py")

# ================================================================
# 5. main.py - final clean version
# ================================================================
main_content = """
import cv2
from camera.webcam                 import Webcam
from object_detection.detector     import Detector
from distance_estimation.estimator import Estimator
from voice.speaker                 import Speaker
from voice.alerts                  import AlertSystem
from tracking.tracker              import Tracker
from utils.draw                    import draw_detection, draw_hud, draw_sidebar
from utils.fps                     import FPSCounter
from utils.helpers                 import save_screenshot
from config                        import WINDOW_TITLE, SHOW_SIDEBAR

def main():
    print("Starting Vision Assistant...")

    # Initialize modules
    camera    = Webcam()
    detector  = Detector()
    estimator = Estimator()
    speaker   = Speaker()
    alerts    = AlertSystem(speaker)
    tracker   = Tracker()
    fps_ctr   = FPSCounter()

    # Start camera
    camera.start()
    speaker.say("Vision Assistant started. Camera is active.")

    paused = False
    frame  = None

    print("\\n Controls:")
    print("   Q     = Quit")
    print("   V     = Toggle Voice ON/OFF")
    print("   S     = Save Screenshot")
    print("   SPACE = Pause/Resume\\n")

    while True:
        if not paused:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Camera read failed - retrying...")
                continue

            # Get frame dimensions
            h, w = frame.shape[:2]

            # Step 1: Detect ALL objects
            detections = detector.detect(frame)

            # Step 2: Estimate distance for each object
            for det in detections:
                estimator.estimate(det, frame_width=w)

            # Step 3: Track objects
            if detections:
                detections = tracker.update(detections)

            # Step 4: Voice alerts for dangerous objects
            alerts.process(detections)

            # Step 5: Draw everything on frame
            # Draw sidebar first (background)
            if SHOW_SIDEBAR:
                draw_sidebar(frame, detections)

            # Draw each detection box
            for det in detections:
                draw_detection(frame, det)

            # Draw HUD last (on top)
            fps = fps_ctr.update()
            draw_hud(frame, fps, len(detections), speaker.enabled)

            # Show frame
            cv2.imshow(WINDOW_TITLE, frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            speaker.say("Shutting down. Goodbye!")
            break
        elif key == ord("v"):
            speaker.toggle()
        elif key == ord("s"):
            if frame is not None:
                save_screenshot(frame)
                print("Screenshot saved!")
        elif key == ord(" "):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"System {status}")

    camera.stop()
    cv2.destroyAllWindows()
    print("System stopped.")

if __name__ == "__main__":
    main()
"""

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_content.lstrip())
print("  Written: main.py")

# ================================================================
# 6. Fix model path and download if needed
# ================================================================
print("\nChecking model...")
import os

os.makedirs("models", exist_ok=True)

model_ok = False
model_path = "models/yolov8n.pt"

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if size_mb > 1:
        print(f"  Model found: {model_path} ({size_mb:.1f} MB) OK")
        model_ok = True
    else:
        print(f"  Model corrupted ({size_mb:.2f} MB) - re-downloading...")
        os.remove(model_path)

if not model_ok:
    print("  Downloading yolov8n.pt...")
    try:
        from ultralytics import YOLO
        import shutil
        model = YOLO("yolov8n.pt")
        if os.path.exists("yolov8n.pt"):
            shutil.move("yolov8n.pt", model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  Downloaded and saved: {model_path} ({size_mb:.1f} MB)")
        else:
            print("  Download location unknown - check models/ folder")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Manual download: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
        print(f"  Place file at: {os.path.abspath(model_path)}")

# ================================================================
# 7. Final test
# ================================================================
print("\n--- Final Import Test ---")
all_ok = True
tests = [
    ("config",                        "from config import CAMERA_INDEX, CONFIDENCE, MAX_DETECTIONS"),
    ("camera.webcam",                 "from camera.webcam import Webcam"),
    ("object_detection.detector",     "from object_detection.detector import Detector"),
    ("distance_estimation.estimator", "from distance_estimation.estimator import Estimator"),
    ("voice.speaker",                 "from voice.speaker import Speaker"),
    ("voice.alerts",                  "from voice.alerts import AlertSystem"),
    ("tracking.tracker",              "from tracking.tracker import Tracker"),
    ("utils.draw",                    "from utils.draw import draw_detection, draw_hud, draw_sidebar"),
    ("utils.fps",                     "from utils.fps import FPSCounter"),
    ("utils.helpers",                 "from utils.helpers import save_screenshot"),
]

for name, stmt in tests:
    try:
        exec(stmt)
        print(f"  OK     {name}")
    except Exception as e:
        print(f"  FAIL   {name} --> {e}")
        all_ok = False

print()
if all_ok:
    print("=" * 50)
    print("ALL OK! Run: python main.py")
    print("=" * 50)
else:
    print("Some imports failed - share output above!")
