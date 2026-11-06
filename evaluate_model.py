# evaluate_model.py
# Live Model Evaluation — Accuracy, Precision & Recall
# Blind Assistant System
#
# Usage:  python evaluate_model.py
# Controls:
#   SPACE  = Capture current frame for evaluation
#   R      = Show running metrics
#   Q      = Finish & show final report

import cv2
import os
import time
import sys
from datetime import datetime
from collections import defaultdict

# ── Import existing pipeline modules ────────────────────────────────
from camera.webcam                 import Webcam
from object_detection.detector     import Detector
from distance_estimation.estimator import Estimator

# ════════════════════════════════════════════════════════════════════
#  Metrics Tracker
# ════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """Tracks TP, FP, FN per class and overall."""

    def __init__(self):
        self.tp = 0          # True Positives  (correct detection)
        self.fp = 0          # False Positives  (wrong detection)
        self.fn = 0          # False Negatives  (missed real object)
        self.total_frames = 0
        self.total_detections = 0

        # Per-class tracking
        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)

    def add_tp(self, cls_name):
        self.tp += 1
        self.class_tp[cls_name] += 1

    def add_fp(self, cls_name):
        self.fp += 1
        self.class_fp[cls_name] += 1

    def add_fn(self, cls_name="unknown"):
        self.fn += 1
        self.class_fn[cls_name] += 1

    @property
    def precision(self):
        denom = self.tp + self.fp
        return (self.tp / denom * 100) if denom > 0 else 0.0

    @property
    def recall(self):
        denom = self.tp + self.fn
        return (self.tp / denom * 100) if denom > 0 else 0.0

    @property
    def accuracy(self):
        """Jaccard-style accuracy: TP / (TP + FP + FN)"""
        denom = self.tp + self.fp + self.fn
        return (self.tp / denom * 100) if denom > 0 else 0.0

    @property
    def f1_score(self):
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def class_precision(self, cls):
        d = self.class_tp[cls] + self.class_fp[cls]
        return (self.class_tp[cls] / d * 100) if d > 0 else 0.0

    def class_recall(self, cls):
        d = self.class_tp[cls] + self.class_fn[cls]
        return (self.class_tp[cls] / d * 100) if d > 0 else 0.0

    def all_classes(self):
        """Return sorted list of all classes that appeared."""
        classes = set()
        classes.update(self.class_tp.keys())
        classes.update(self.class_fp.keys())
        classes.update(self.class_fn.keys())
        return sorted(classes)


# ════════════════════════════════════════════════════════════════════
#  Drawing helpers for evaluation mode
# ════════════════════════════════════════════════════════════════════

LEVEL_COLORS = {
    "DANGER":   (0,   0,   255),
    "MODERATE": (0,  165,  255),
    "CAUTION":  (0,  255,  255),
    "SAFE":     (0,  255,    0),
    "UNKNOWN":  (180, 180,  180),
}


def draw_numbered_detections(frame, detections):
    """Draw each detection with a big numbered circle for easy identification."""
    annotated = frame.copy()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        level = det.get("danger_level", "UNKNOWN")
        color = LEVEL_COLORS.get(level, (255, 255, 255))
        conf  = det.get("confidence", 0)
        cls   = det.get("class", "object")
        dist  = det.get("distance", None)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Numbered circle (top-left of box)
        num = idx + 1
        cx, cy = x1 + 15, y1 + 15
        cv2.circle(annotated, (cx, cy), 16, color, -1)
        cv2.circle(annotated, (cx, cy), 16, (0, 0, 0), 2)
        cv2.putText(annotated, str(num), (cx - 7, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Label with class, distance, confidence
        dist_txt = f"{dist:.1f}m" if dist else "?m"
        label = f"#{num} {cls} | {dist_txt} | {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(annotated, (x1, label_y - th - 4),
                      (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y),
                    font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Instruction bar at bottom
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(annotated, "EVALUATION MODE | SPACE=Capture  R=Metrics  Q=Finish",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 1, cv2.LINE_AA)

    return annotated


def draw_live_feed(frame, det_count, metrics):
    """Draw minimal HUD on live feed (before capture)."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Top bar
    cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.putText(display, "LIVE EVALUATION MODE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 200, 255), 2, cv2.LINE_AA)

    info = f"Detections: {det_count} | Frames evaluated: {metrics.total_frames}"
    cv2.putText(display, info, (w - 500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Metrics overlay (if we have data)
    if metrics.total_frames > 0:
        y = 75
        stats = [
            f"Precision: {metrics.precision:.1f}%",
            f"Recall:    {metrics.recall:.1f}%",
            f"Accuracy:  {metrics.accuracy:.1f}%",
            f"F1-Score:  {metrics.f1_score:.1f}%",
        ]
        cv2.rectangle(display, (5, 50), (230, 50 + len(stats) * 28 + 10),
                       (20, 20, 20), -1)
        for s in stats:
            cv2.putText(display, s, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 1, cv2.LINE_AA)
            y += 28

    # Bottom instruction bar
    cv2.rectangle(display, (0, h - 40), (w, h), (30, 30, 30), -1)
    cv2.putText(display, "SPACE = Capture & Evaluate  |  R = Show Metrics  |  Q = Finish & Report",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return display


# ════════════════════════════════════════════════════════════════════
#  Report generation
# ════════════════════════════════════════════════════════════════════

def generate_report(metrics):
    """Build the final report string."""
    sep  = "=" * 55
    sep2 = "-" * 55
    lines = []

    lines.append("")
    lines.append(sep)
    lines.append("  BLIND ASSISTANT SYSTEM — EVALUATION REPORT")
    lines.append(sep)
    lines.append(f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model:             YOLOv8n (COCO pre-trained)")
    lines.append(f"  Frames Evaluated:  {metrics.total_frames}")
    lines.append(f"  Total Detections:  {metrics.total_detections}")
    lines.append(sep2)
    lines.append(f"  True Positives:    {metrics.tp}")
    lines.append(f"  False Positives:   {metrics.fp}")
    lines.append(f"  False Negatives:   {metrics.fn}")
    lines.append(sep2)
    lines.append(f"  PRECISION:  {metrics.precision:.1f}%")
    lines.append(f"  RECALL:     {metrics.recall:.1f}%")
    lines.append(f"  ACCURACY:   {metrics.accuracy:.1f}%  (Jaccard)")
    lines.append(f"  F1-SCORE:   {metrics.f1_score:.1f}%")
    lines.append(sep2)

    # Per-class breakdown
    classes = metrics.all_classes()
    if classes:
        lines.append("")
        lines.append("  Per-Class Breakdown:")
        lines.append(f"  {'Class':<18} {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>7}  {'Recall':>7}")
        lines.append("  " + "-" * 52)
        for cls in classes:
            tp = metrics.class_tp[cls]
            fp = metrics.class_fp[cls]
            fn = metrics.class_fn[cls]
            p  = metrics.class_precision(cls)
            r  = metrics.class_recall(cls)
            lines.append(f"  {cls:<18} {tp:>4}  {fp:>4}  {fn:>4}  {p:>6.1f}%  {r:>6.1f}%")

    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def print_running_metrics(metrics):
    """Print a compact metrics summary to console."""
    print("\n" + "-" * 40)
    print(f"  Running Metrics ({metrics.total_frames} frames)")
    print("-" * 40)
    print(f"  TP: {metrics.tp}  |  FP: {metrics.fp}  |  FN: {metrics.fn}")
    print(f"  Precision: {metrics.precision:.1f}%")
    print(f"  Recall:    {metrics.recall:.1f}%")
    print(f"  Accuracy:  {metrics.accuracy:.1f}%")
    print(f"  F1-Score:  {metrics.f1_score:.1f}%")
    print("-" * 40 + "\n")


# ════════════════════════════════════════════════════════════════════
#  Interactive frame evaluation
# ════════════════════════════════════════════════════════════════════

def evaluate_frame(frame, detections, metrics, frame_num):
    """
    Show annotated frame and let user verify each detection.
    Returns updated metrics.
    """
    num_dets = len(detections)
    metrics.total_detections += num_dets

    print(f"\n{'='*50}")
    print(f"  FRAME #{frame_num} — {num_dets} detection(s) found")
    print(f"{'='*50}")

    # Show the numbered detections in OpenCV window
    annotated = draw_numbered_detections(frame, detections)
    cv2.imshow("Evaluation - Verify Detections", annotated)
    cv2.waitKey(1)

    # ── Verify each detection ───────────────────────────────────
    if num_dets > 0:
        print("\n  For each detection, type:")
        print("    y = Correct (True Positive)")
        print("    n = Wrong   (False Positive)")
        print("    s = Skip    (counts as correct)\n")

        for idx, det in enumerate(detections):
            cls  = det.get("class", "object")
            conf = det.get("confidence", 0)
            dist = det.get("distance", None)
            level = det.get("danger_level", "UNKNOWN")
            dist_txt = f"{dist:.1f}m" if dist else "?m"

            prompt = f"  #{idx+1}  {cls:<15} {dist_txt:<8} {conf:.0%}  {level:<10} -- Correct? (y/n/s): "

            while True:
                answer = input(prompt).strip().lower()
                if answer in ("y", "yes", "s", "skip", ""):
                    metrics.add_tp(cls)
                    print(f"         >> Marked as TRUE POSITIVE")
                    break
                elif answer in ("n", "no"):
                    metrics.add_fp(cls)
                    print(f"         >> Marked as FALSE POSITIVE")
                    break
                else:
                    print("         Please enter y, n, or s")
    else:
        print("  No detections in this frame.")

    # ── Ask about missed objects (False Negatives) ──────────────
    print(f"\n  Were there objects the model MISSED in this frame?")
    while True:
        missed_input = input("  Number of missed objects (0 if none): ").strip()
        try:
            missed_count = int(missed_input)
            if missed_count < 0:
                print("  Please enter 0 or a positive number.")
                continue
            break
        except ValueError:
            print("  Please enter a number.")

    if missed_count > 0:
        print(f"  Enter the class name for each missed object (or press Enter for 'unknown'):")
        for i in range(missed_count):
            cls_input = input(f"    Missed #{i+1} class name: ").strip().lower()
            if not cls_input:
                cls_input = "unknown"
            metrics.add_fn(cls_input)
            print(f"         >> Marked as FALSE NEGATIVE ({cls_input})")

    metrics.total_frames += 1

    # Show quick frame summary
    print(f"\n  Frame #{frame_num} done. Running totals:")
    print(f"  TP={metrics.tp}  FP={metrics.fp}  FN={metrics.fn}")
    print(f"  Precision={metrics.precision:.1f}%  Recall={metrics.recall:.1f}%  F1={metrics.f1_score:.1f}%")


# ════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  BLIND ASSISTANT SYSTEM — LIVE MODEL EVALUATION")
    print("=" * 55)
    print("  Model:     YOLOv8n (COCO pre-trained)")
    print("  Pipeline:  Detection -> Distance -> Danger Level")
    print("-" * 55)
    print("  Controls:")
    print("    SPACE  = Capture frame & start evaluation")
    print("    R      = Show running metrics")
    print("    Q      = Finish & generate report")
    print("=" * 55 + "\n")

    # Initialize pipeline
    print("  Initializing modules...")
    camera    = Webcam()
    detector  = Detector()
    estimator = Estimator()
    metrics   = MetricsTracker()
    frame_num = 0

    camera.start()
    print("  Camera ready. Point at objects and press SPACE to evaluate.\n")

    # Create output directory
    os.makedirs("evaluation_results/captured_frames", exist_ok=True)

    eval_window = "Evaluation - Verify Detections"
    running = True

    while running:
        ret, frame = camera.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        h, w = frame.shape[:2]

        # Run detection pipeline (continuous — so user sees detections live)
        detections = detector.detect(frame)
        estimator.estimate(detections, frame_width=w)

        # Draw live feed with HUD
        display = draw_live_feed(frame, len(detections), metrics)

        # Draw detection boxes on live view
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            level = det.get("danger_level", "UNKNOWN")
            color = LEVEL_COLORS.get(level, (255, 255, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            cls = det.get("class", "?")
            dist = det.get("distance")
            dist_txt = f"{dist:.1f}m" if dist else "?m"
            label = f"{cls} {dist_txt}"
            cv2.putText(display, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow("Live Evaluation", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("\n  Finishing evaluation...\n")
            running = False

        elif key == ord("r"):
            print_running_metrics(metrics)

        elif key == ord(" "):
            # ── Capture & Evaluate ──────────────────────────────
            frame_num += 1

            # Re-run detection on the captured frame (frozen)
            captured_frame = frame.copy()
            captured_dets  = detector.detect(captured_frame)
            estimator.estimate(captured_dets, frame_width=w)

            # Save captured frame
            frame_path = f"evaluation_results/captured_frames/frame_{frame_num:03d}.jpg"
            cv2.imwrite(frame_path, captured_frame)

            # Interactive evaluation (blocks until user finishes)
            evaluate_frame(captured_frame, captured_dets, metrics, frame_num)

            # Close evaluation window after frame is done
            cv2.destroyWindow(eval_window)
            print(f"\n  Press SPACE for next frame, R for metrics, Q to finish.\n")

    # ── Cleanup ─────────────────────────────────────────────────
    camera.stop()
    cv2.destroyAllWindows()

    # ── Final Report ────────────────────────────────────────────
    if metrics.total_frames > 0:
        report = generate_report(metrics)
        print(report)

        # Save report to file
        report_path = "evaluation_results/report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved to: {os.path.abspath(report_path)}")
    else:
        print("\n  No frames were evaluated. Exiting.\n")

    print("\n  Evaluation complete.\n")


if __name__ == "__main__":
    main()
