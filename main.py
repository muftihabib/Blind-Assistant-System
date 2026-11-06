import cv2
import time
import sys

from camera.webcam                  import Webcam
from object_detection.detector      import Detector
from distance_estimation.estimator  import Estimator
from voice.speaker                  import Speaker
from utils.draw                     import draw_detections, draw_hud
from utils.fps                      import FPSCounter


def main():
    print("=" * 50)
    print("  Blind Assistance System  Starting Up")
    print("=" * 50)

    # ── Initialise all subsystems ────────────────────────────────────────
    try:
        cam = Webcam().start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    det = Detector()
    est = Estimator()
    spk = Speaker()
    fps = FPSCounter()

    print("[INFO] System ready.  Q=Quit   V=Voice toggle")

    SKIP            = 3       # detect every 3rd frame (balance speed + accuracy)
    fc              = 0
    last_detections = []

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        fc += 1
        fps.update()

        # ── Detection + estimation every SKIP frames ─────────────────────
        if fc % SKIP == 0:
            detections      = det.detect(frame)
            last_detections = est.estimate(detections, frame.shape[1])
            # ↑ each detection now has: distance, danger_level, position, nav_direction

            # ── Speak alerts & navigation guidance ───────────────────────
            spk.speak(last_detections)

        # ── Render ───────────────────────────────────────────────────────
        display = frame.copy()
        display = draw_detections(display, last_detections)
        display = draw_hud(display, fps.get(), len(last_detections), last_detections)

        cv2.imshow("Blind Assistance System  |  Q=Quit  V=Voice", display)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:      # Q or Esc → quit
            break
        elif key == ord("v"):                  # V → toggle voice
            spk.toggle()

    # ── Cleanup ──────────────────────────────────────────────────────────
    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")


if __name__ == "__main__":
    main()
