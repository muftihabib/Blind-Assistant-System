# Blind Assistance System

Real-time obstacle detection with monocular distance estimation and offline voice alerts,
running on a CPU with an ordinary webcam.

```
frame ─► downscale + denoise + CLAHE ─► YOLOv8n ─► pinhole distance (D = W_r·f / w_b)
      ─► centroid tracking + 5-frame smoothing ─► danger level ─► path guidance
      ─► prioritised alert (3 s cooldown per object, approach override) ─► pyttsx3 (offline)
```

## Setup

```bash
pip install -r requirements.txt
python -m distance_estimation.calibration checkerboard --cols 9 --rows 6 --square-mm 25
python main.py
```

Calibrate first. Without `calibration/camera_intrinsics.json` the system falls back to
`FOCAL_LENGTH` in `config.py` and prints a warning. Do not report results with the fallback.

Keys: `Q` quit, `V` voice on/off, `S` screenshot.

## Layout

| Path | Paper section | What it does |
|---|---|---|
| `config.py` | all | Every threshold, prior and timing in one place |
| `object_detection/preprocess.py` | III-B | 3×3 Gaussian denoise + CLAHE (L channel, clip 2.0, 8×8) |
| `object_detection/detector.py` | III-C | YOLOv8n. Boxes are returned in **capture-frame pixels** |
| `distance_estimation/calibration.py` | III-E.1 | Checkerboard `calibrateCamera`, single-shot check, intrinsics loading |
| `distance_estimation/estimator.py` | III-D/E | Pinhole distance, side-profile width switch, clipping, occlusion, danger levels |
| `tracking/tracker.py` | III-F | Centroid tracker with per-class matching and 5-frame moving-average distance |
| `navigation/guidance.py` | III-G | Clear/Blocked sectors and the full steering rule set |
| `voice/alerts.py` | III-H | Priority, per-track cooldown, approach / enter-DANGER override |
| `voice/speaker.py` | III-H | pyttsx3 worker (SAPI fallback) with pre-emption and latency logging |
| `tools/eval_benchmark.py` | V-B/C | P/R/F1/AP@0.5, miss analysis, distance error by zone and class |
| `tools/summarize_alerts.py` | V-D | Alerts/min, time to first audio / end of utterance, reaction margin |
| `tests/test_pipeline.py` | — | Unit tests for the decision logic |

## Key definitions

**Coordinate frame.** Ultralytics returns boxes in the frame passed to it. The detector downscales the
1280×720 capture to 640 px wide, runs the model, and scales boxes back. As a result, `w_b` is always in
1280×720 capture pixels, and the focal length must be calibrated at that resolution.
`load_intrinsics()` rescales automatically if the resolutions differ. Lens distortion is corrected
on the two box edge points (`cv2.undistortPoints`) before the width is measured.

**Near-field clipping.** If a box touches the left or right frame edge, `w_b` is truncated and the
pinhole distance is only an *upper bound* (`clipped = True`; shown as `<1.2m`, spoken as "very close" or
"within N meters"). The closest distance at which the whole prior fits is `D_min = W_r·f / frame_width`.
A clipped box covering ≥ 50 % of the frame width, or touching both edges, is forced to DANGER.

**Width priors.** Frontal widths are in `KNOWN_WIDTHS`. Bicycle, motorcycle and car switch to the side
width (175 / 175 / 420 cm) when `w_b/h_b` exceeds 1.2 / 1.2 / 1.4. Person has no side prior, because a
person's box aspect ratio cannot tell frontal from side-on.

**Guidance.** A box *occupies* a sector if it covers ≥ 10 % of that sector's width.

- **Hazard:** something occupying CENTER at ≤ 2.5 m.
- **Clear:** a side is Clear if everything occupying it is > 3.0 m away; otherwise it is Blocked.
- **Rules:**
  - No hazard → no cue.
  - Only left Clear → move left.
  - Only right Clear → move right.
  - Both Clear → the side with more clearance.
  - Both Blocked → "stop, path blocked".

**Alerts.** Only DANGER, WARNING and CAUTION are spoken.

- **Priority:** (level, people and vehicles first, distance).
- **Cooldown:** one alert per track per 3 s.
- **Override:** the cooldown is skipped if the object is ≥ 0.5 m closer than at its last alert, or has
  just entered DANGER. Such an alert is an *emergency* and interrupts ordinary (non-emergency) speech.
- **Acceptance:** the speaker only accepts an alert it can start immediately; otherwise the alert stays
  eligible for the next frame. No backlog builds up, and no alert is marked "announced" without being
  spoken.

## Experiments

```bash
# Detection + distance benchmark (same protocol for every detector)
python -m tools.eval_benchmark --data bench/ --model models/yolov8n.pt --name yolov8n
python -m tools.eval_benchmark --data bench/ --model yolov5nu.pt     --name yolov5n
python -m tools.eval_benchmark --data bench/ --stride 10   # use every 10th frame (less correlated)

# Alert-management ablation on the same recorded video
python main.py --source scene.mp4 --log logs/full.csv
python main.py --source scene.mp4 --log logs/naive.csv --no-tracking --no-cooldown --fifo
python -m tools.summarize_alerts logs/full.csv logs/naive.csv

# Pre-processing ablation
python -m tools.eval_benchmark --data bench/ --no-preprocess --name yolov8n_nopre
```

**Benchmark labels:** YOLO format, one `.txt` per image, one object per line:
`class_id cx cy w h [distance_m]`, where `class_id` is the COCO index and `distance_m` is the laser
ground truth. `results/<name>/metrics.json` contains:

- the full protocol
- P/R/F1/mAP@0.5, overall and per class
- recall by distance zone
- a miss analysis (`below_conf_threshold`, `wrong_class`, `poor_localisation`, `not_detected`)
- distance accuracy (±15 %), MAE, MSE, RMSE, bias and relative error, by zone, by class and by
  class × zone, with and without clipped boxes

**Alert log:** every alert is logged to `logs/alerts.csv` with capture → queued → audio start → audio end
timestamps. The audio timestamps are SAPI stream events and include the engine's leading and trailing
silence plus device buffering (about 1.2 s even for a one-word utterance), so treat them as upper bounds.

## Tests

```bash
python -m pytest -q tests
```
