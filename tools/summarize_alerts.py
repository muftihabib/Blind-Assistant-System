"""
Summarise an alert log written by voice/speaker.py.

    python -m tools.summarize_alerts logs/alerts.csv
    python -m tools.summarize_alerts logs/full.csv logs/naive.csv

Reports alerts/min, time-to-first-audio and time-to-end-of-utterance
(median, p95, max), utterance length, interrupted utterances, and — for
emergency alerts — the reaction margin left at a given approach speed:

    margin_first_audio = (D_alert - D_collision) / v  - time_to_first_audio
    margin_end         = (D_alert - D_collision) / v  - time_to_end

If a run summary (logs/x_run.json, written by main.py) exists next to the log,
its duration is used for alerts/min; otherwise the span of logged alerts is used.
"""

import argparse
import csv
import json
import os

import numpy as np


def _pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def alert_stats(path, speed=1.1, collision=0.3):
    """Return a dict of alert statistics for one log (None if the log is empty/missing)."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    col = lambda k, rs=rows: np.array([float(r[k]) for r in rs])

    run_path = os.path.splitext(path)[0] + "_run.json"
    run = json.load(open(run_path)) if os.path.exists(run_path) else None
    t = col("t_capture")
    minutes = (run["elapsed_s"] / 60.0) if run else max((t.max() - t.min()) / 60.0, 1e-9)

    tfa, tend, utt = col("time_to_first_audio_ms"), col("time_to_end_ms"), col("utterance_ms")
    s = {"log": path, "n_alerts": len(rows), "minutes": round(minutes, 2),
         "alerts_per_min": round(len(rows) / minutes, 1),
         "interrupted": sum(r["completed"] != "True" for r in rows),
         "first_audio_ms_median": _pct(tfa, 50), "first_audio_ms_p95": _pct(tfa, 95),
         "first_audio_ms_max": round(float(tfa.max()), 1),
         "end_ms_median": _pct(tend, 50), "end_ms_p95": _pct(tend, 95),
         "end_ms_max": round(float(tend.max()), 1),
         "utterance_ms_median": _pct(utt, 50),
         "fps": run["fps"] if run else None, "run": run}

    em = [r for r in rows if r["emergency"] == "True"]
    s["n_emergency"] = len(em)
    if em:
        d = col("distance_m", em)
        avail = (d - collision) / speed
        m1 = avail - col("time_to_first_audio_ms", em) / 1000
        m2 = avail - col("time_to_end_ms", em) / 1000
        s.update({"emergency_mean_distance_m": round(float(d.mean()), 2),
                  "time_available_s": round(float(avail.mean()), 2),
                  "margin_first_audio_s_mean": round(float(m1.mean()), 2),
                  "margin_first_audio_s_min": round(float(m1.min()), 2),
                  "margin_end_s_mean": round(float(m2.mean()), 2),
                  "margin_end_s_min": round(float(m2.min()), 2),
                  "speed_mps": speed, "collision_m": collision})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--speed", type=float, default=1.1, help="approach speed m/s")
    ap.add_argument("--collision", type=float, default=0.3, help="collision perimeter m")
    a = ap.parse_args()
    for p in a.logs:
        s = alert_stats(p, a.speed, a.collision)
        print(f"\n== {p} ==")
        if s is None:
            print("  empty or missing"); continue
        print(f"alerts: {s['n_alerts']} over {s['minutes']} min -> {s['alerts_per_min']}/min "
              f"(interrupted: {s['interrupted']})" + (f", {s['fps']} FPS" if s["fps"] else ""))
        print(f"time to first audio  ms: median {s['first_audio_ms_median']}  "
              f"p95 {s['first_audio_ms_p95']}  max {s['first_audio_ms_max']}")
        print(f"time to end of speech ms: median {s['end_ms_median']}  "
              f"p95 {s['end_ms_p95']}  max {s['end_ms_max']}")
        print(f"utterance length     ms: median {s['utterance_ms_median']}")
        if s["n_emergency"]:
            print(f"emergency alerts: {s['n_emergency']}, mean alert distance "
                  f"{s['emergency_mean_distance_m']} m")
            print(f"  reaction margin @ {a.speed} m/s to {a.collision} m: after first audio "
                  f"{s['margin_first_audio_s_mean']}s (min {s['margin_first_audio_s_min']}), "
                  f"after end of utterance {s['margin_end_s_mean']}s (min {s['margin_end_s_min']})")


if __name__ == "__main__":
    main()
