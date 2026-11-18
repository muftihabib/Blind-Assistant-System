"""
Blind Assistance System — real-time pipeline.

    frame -> pre-process -> YOLOv8n -> pinhole distance -> centroid tracking
          -> 5-frame smoothing -> danger level -> path guidance
          -> prioritised alert (cooldown + approach override) -> offline TTS

Usage
    python main.py                         # webcam
    python main.py --source walk.mp4       # repeatable test on a video file
    python main.py --no-tracking --no-cooldown --fifo   # naive baseline for the ablation
    python main.py --no-preprocess         # pre-processing ablation

Keys: Q / Esc = quit, V = voice on/off, S = screenshot
"""

import argparse
import json
import os
import sys
import time

import cv2

from camera.webcam import Webcam
from config import ALERT_LOG_FILE, CAMERA_INDEX, DETECT_EVERY_N
from distance_estimation.estimator import Estimator
from navigation.guidance import compute_guidance
from object_detection.detector import Detector
from tracking.tracker import Tracker
from utils.draw import draw_detections, draw_hud
from utils.fps import FPSCounter
from voice.alerts import AlertManager
from voice.speaker import Speaker


def parse_args():
    ap = argparse.ArgumentParser(description="Blind Assistance System")
    ap.add_argument("--source", default=str(CAMERA_INDEX), help="camera index or video file")
    ap.add_argument("--model", default=None, help="detector weights (default: config MODEL_PATH)")
    ap.add_argument("--no-tracking", action="store_true", help="ablation: no tracking / smoothing")
    ap.add_argument("--no-cooldown", action="store_true", help="ablation: announce every eligible frame")
    ap.add_argument("--fifo", action="store_true",
                    help="ablation: naive FIFO speech queue (every alert queued, no pre-emption)")
    ap.add_argument("--no-preprocess", action="store_true", help="ablation: skip denoise + CLAHE")
    ap.add_argument("--tts", choices=["pyttsx3", "sapi"], default=None)
    ap.add_argument("--log", default=None, help="alert log CSV (default: config ALERT_LOG_FILE)")
    ap.add_argument("--headless", action="store_true", help="no display window")
    return ap.parse_args()


def main():
    args = parse_args()
    print("=" * 50)
    print("  Blind Assistance System  starting")
    print("=" * 50)

    try:
        cam = Webcam(args.source).start()
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")

    det = Detector(args.model, preprocess=not args.no_preprocess)
    est = Estimator()
    trk = Tracker(enabled=not args.no_tracking)
    alerts = AlertManager(use_cooldown=not args.no_cooldown)
    spk_kwargs = {}
    if args.tts:
        spk_kwargs["backend"] = args.tts
    if args.log:
        spk_kwargs["log_file"] = args.log
    spk = Speaker(fifo=args.fifo, **spk_kwargs)
    fps = FPSCounter()

    print(f"[INFO] tracking={'off' if args.no_tracking else 'on'}  "
          f"cooldown={'off' if args.no_cooldown else 'on'}  "
          f"preprocess={'off' if args.no_preprocess else 'on'}  "
          f"speech queue={'FIFO' if args.fifo else 'single-slot'}")
    print("[INFO] Q=quit  V=voice  S=screenshot")

    fc, last_t = 0, None
    dets, guidance = [], {"action": "CLEAR", "message": None, "hazard": None}
    n_alerts, t0 = 0, time.perf_counter()

    while True:
        ok, frame, t_cap = cam.read()
        if not ok:
            if cam.ended:
                break
            time.sleep(0.005)
            continue
        if t_cap == last_t:               # same webcam frame as last iteration
            time.sleep(0.002)
            continue
        last_t = t_cap
        fc += 1
        fps.update()
        h, w = frame.shape[:2]

        if fc % DETECT_EVERY_N == 0:
            dets = det.detect(frame)
            est.estimate(dets, w, h)
            trk.update(dets, w)
            est.finalize(dets, w)
            guidance = compute_guidance(dets, w)

            alert = alerts.select(dets, guidance, t_cap)
            if alert is not None and spk.offer(alert):
                alerts.commit(alert, dets)
                n_alerts += 1
            alerts.prune(trk.active_ids if trk.enabled else {d["track_id"] for d in dets})

        if not args.headless:
            display = draw_detections(frame.copy(), dets)
            display = draw_hud(display, fps.get(), len(dets), dets, guidance)
            cv2.imshow("Blind Assistance System  |  Q=Quit  V=Voice", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                spk.toggle()
            if key == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                cv2.imwrite(f"screenshots/capture_{int(time.time())}.jpg", display)

    elapsed = time.perf_counter() - t0
    cam.stop()
    cv2.destroyAllWindows()
    # let the last utterance finish so it is logged
    t_wait = time.perf_counter()
    while spk.busy and time.perf_counter() - t_wait < 5:
        time.sleep(0.05)
    spk.close()
    print(f"[INFO] {fc} frames in {elapsed:.1f}s ({fc / max(elapsed, 1e-9):.1f} FPS), "
          f"{n_alerts} alerts ({60 * n_alerts / max(elapsed, 1e-9):.1f}/min)")

    # Run summary next to the alert log (logs/x.csv -> logs/x_run.json), read by paper_tables/
    summary_path = os.path.splitext(args.log or ALERT_LOG_FILE)[0] + "_run.json"
    with open(summary_path, "w") as f:
        json.dump({"source": args.source, "frames": fc, "elapsed_s": round(elapsed, 2),
                   "fps": round(fc / max(elapsed, 1e-9), 2), "alerts_accepted": n_alerts,
                   "alerts_per_min": round(60 * n_alerts / max(elapsed, 1e-9), 2),
                   "tracking": not args.no_tracking, "cooldown": not args.no_cooldown,
                   "preprocess": not args.no_preprocess, "fifo": args.fifo,
                   "detect_every_n": DETECT_EVERY_N}, f, indent=2)
    print(f"[INFO] run summary -> {summary_path}")


if __name__ == "__main__":
    main()
