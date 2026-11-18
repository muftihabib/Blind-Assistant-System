"""
Reproducible benchmark: detection (P/R/F1/AP@0.5), miss analysis and
monocular distance error — one fixed protocol for every detector.

Dataset layout (YOLO format, one .txt per image, same stem):
    images/  frame_0001.jpg ...
    labels/  frame_0001.txt     lines:  class_id cx cy w h [distance_m]
             class_id = COCO index (0 person, 1 bicycle, 2 car, 3 motorcycle,
             39 bottle, 56 chair, ...); cx cy w h normalised to [0,1];
             distance_m = laser ground truth along the optical axis (optional).

Protocol (printed and saved with the results):
    identical to live use (Detector.predict_raw): capture frame -> downscale to
    --imgsz wide -> denoise + CLAHE (unless --no-preprocess) -> letterbox to
    --imgsz -> NMS IoU --nms-iou, max 300 det -> boxes in capture pixels.
    P/R/F1 at confidence >= --conf; AP@0.5 from all predictions >= 0.001
    (all-point interpolation); match = same class, IoU >= --match-iou, greedy
    by confidence. Predictions of classes not in --classes are ignored.

Examples
    python -m tools.eval_benchmark --data bench/ --model models/yolov8n.pt
    python -m tools.eval_benchmark --data bench/ --model yolov5nu.pt --name yolov5n
    python -m tools.eval_benchmark --data bench/ --stride 10   # decorrelate video frames
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from config import CONFIDENCE, IMGSZ, IOU_THRESHOLD
from distance_estimation.estimator import Estimator, danger_level, iou
from tools.backends import SsdBackend, YoloBackend

DEFAULT_CLASSES = "person,bicycle,motorcycle,car,chair,bottle"
ZONES = ("DANGER", "WARNING", "CAUTION", "SAFE", "no_gt_distance")


def read_labels(path, w, h):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            dist = float(p[5]) if len(p) > 5 else None
            out.append({"cls_id": c, "dist": dist,
                        "bbox": ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h)})
    return out


def average_precision(confs, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if not confs:
        return 0.0
    order = np.argsort(-np.asarray(confs))
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def dist_stats(rows):
    if not rows:
        return {"n": 0}
    e = np.array([r["est"] - r["gt"] for r in rows])
    rel = np.abs(e) / np.array([r["gt"] for r in rows])
    lvl_ok = np.mean([danger_level(r["est"]) == danger_level(r["gt"]) for r in rows])
    return {"n": len(rows),
            "acc_within_15pct": round(100 * float(np.mean(rel <= 0.15)), 1),
            "mae_m": round(float(np.mean(np.abs(e))), 3),
            "mse_m2": round(float(np.mean(e ** 2)), 3),
            "rmse_m": round(float(math.sqrt(np.mean(e ** 2))), 3),
            "bias_m": round(float(np.mean(e)), 3),
            "mean_rel_err_pct": round(100 * float(np.mean(rel)), 1),
            "median_rel_err_pct": round(100 * float(np.median(rel)), 1),
            "danger_level_agreement_pct": round(100 * float(lvl_ok), 1),
            "clipped": int(sum(r["clipped"] for r in rows))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with images/ and labels/")
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default=None, help="results sub-folder name")
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--conf", type=float, default=CONFIDENCE)
    ap.add_argument("--nms-iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth image (sorted by name)")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--out", default="results", help="results root folder")
    ap.add_argument("--backend", choices=["yolo", "ssd"], default="yolo")
    ap.add_argument("--ssd-pb", help="SSD frozen_inference_graph.pb (backend ssd)")
    ap.add_argument("--ssd-pbtxt", help="OpenCV graph text for the SSD (backend ssd)")
    ap.add_argument("--max-images", type=int, default=0, help="limit images (0 = all)")
    a = ap.parse_args(argv)

    if a.backend == "ssd":
        backend = SsdBackend(a.ssd_pb, a.ssd_pbtxt, preprocess=not a.no_preprocess)
        a.model = a.model or os.path.basename(a.ssd_pb)
    else:
        backend = YoloBackend(a.model, not a.no_preprocess, a.nms_iou, a.imgsz)
    names = backend.names
    keep = {c.strip() for c in a.classes.split(",") if c.strip()}
    keep_ids = {i for i, n in names.items() if n in keep}
    est = Estimator()

    imgs = sorted(glob.glob(os.path.join(a.data, "images", "*.jpg")) +
                  glob.glob(os.path.join(a.data, "images", "*.png")))[::a.stride]
    if a.max_images:
        imgs = imgs[:a.max_images]
    if not imgs:
        sys.exit(f"No images in {a.data}/images")

    ap_data = defaultdict(lambda: {"conf": [], "tp": []})
    n_gt = defaultdict(int)
    cnt = {"tp": 0, "fp": 0, "fn": 0}
    cls_cnt = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    zone_cnt = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0})
    miss_reason = defaultdict(lambda: defaultdict(int))
    dist_rows = []

    for k, ip in enumerate(imgs, 1):
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        gts = [g for g in read_labels(os.path.join(a.data, "labels",
                                                   os.path.splitext(os.path.basename(ip))[0] + ".txt"), w, h)
               if g["cls_id"] in keep_ids]
        preds = [{"bbox": b, "conf": c, "cls_id": ci}
                 for b, c, ci in backend.predict(img, 0.001) if ci in keep_ids]
        preds.sort(key=lambda p: -p["conf"])

        for g in gts:
            n_gt[g["cls_id"]] += 1

        # AP matching (all predictions) and operating-point matching (conf >= a.conf)
        for subset, record_ap in ((preds, True), ([p for p in preds if p["conf"] >= a.conf], False)):
            used = set()
            for p in subset:
                best, bi = 0.0, -1
                for gi, g in enumerate(gts):
                    if gi in used or g["cls_id"] != p["cls_id"]:
                        continue
                    o = iou(p["bbox"], g["bbox"])
                    if o > best:
                        best, bi = o, gi
                hit = best >= a.match_iou
                if hit:
                    used.add(bi)
                if record_ap:
                    ap_data[p["cls_id"]]["conf"].append(p["conf"])
                    ap_data[p["cls_id"]]["tp"].append(1 if hit else 0)
                    continue
                cname = names[p["cls_id"]]
                if hit:
                    cnt["tp"] += 1; cls_cnt[cname]["tp"] += 1
                    g = gts[bi]
                    if g["dist"] is not None:
                        x1, y1, x2, y2 = p["bbox"]
                        d = {"class": cname, "bbox": (x1, y1, x2, y2),
                             "bbox_width": x2 - x1, "bbox_height": y2 - y1}
                        est.estimate([d], w, h)
                        dist_rows.append({"cls": cname, "gt": g["dist"], "est": d["distance_raw"],
                                          "clipped": d["clipped"], "view": d["width_view"],
                                          "zone": danger_level(g["dist"])})
                else:
                    cnt["fp"] += 1; cls_cnt[cname]["fp"] += 1
            if record_ap:
                continue
            for gi, g in enumerate(gts):
                cname = names[g["cls_id"]]
                zone = danger_level(g["dist"]) if g["dist"] is not None else "no_gt_distance"
                zone_cnt[zone]["gt"] += 1
                if gi in used:
                    zone_cnt[zone]["tp"] += 1
                    continue
                cnt["fn"] += 1; cls_cnt[cname]["fn"] += 1; zone_cnt[zone]["fn"] += 1
                # Why was it missed?
                any_low = max((iou(p["bbox"], g["bbox"]) for p in preds
                               if p["cls_id"] == g["cls_id"]), default=0.0)
                wrong_cls = max((iou(p["bbox"], g["bbox"]) for p in preds
                                 if p["conf"] >= a.conf and p["cls_id"] != g["cls_id"]), default=0.0)
                low_conf = max((iou(p["bbox"], g["bbox"]) for p in preds
                                if p["conf"] < a.conf and p["cls_id"] == g["cls_id"]), default=0.0)
                if low_conf >= a.match_iou:
                    reason = "below_conf_threshold"
                elif wrong_cls >= a.match_iou:
                    reason = "wrong_class"
                elif any_low >= 0.1:
                    reason = "poor_localisation"
                else:
                    reason = "not_detected"
                miss_reason[f"{zone}"][reason] += 1
                miss_reason[f"class:{cname}"][reason] += 1
        if k % 50 == 0:
            print(f"  {k}/{len(imgs)}")

    P, R, F = prf(cnt["tp"], cnt["fp"], cnt["fn"])
    aps = {names[c]: average_precision(ap_data[c]["conf"], ap_data[c]["tp"], n_gt[c]) for c in n_gt}
    m_ap = float(np.nanmean(list(aps.values()))) * 100 if aps else float("nan")

    by_zone = defaultdict(list); by_cls = defaultdict(list); by_cls_zone = defaultdict(list)
    for r in dist_rows:
        by_zone[r["zone"]].append(r); by_cls[r["cls"]].append(r)
        by_cls_zone[f"{r['cls']}/{r['zone']}"].append(r)

    out = {
        "protocol": {"model": a.model or "config MODEL_PATH", "backend": a.backend,
                     "input": backend.input_desc, "data": os.path.abspath(a.data),
                     "images": len(imgs), "stride": a.stride,
                     "imgsz": a.imgsz, "conf": a.conf, "nms_iou": a.nms_iou, "match_iou": a.match_iou,
                     "preprocess": not a.no_preprocess, "classes": sorted(keep),
                     "focal_px": round(est.focal, 1) if est.intr else None,
                     "intrinsics": est.intr.describe() if est.intr else None},
        "detection": {"precision": round(P, 1), "recall": round(R, 1), "f1": round(F, 1),
                      "mAP50": round(m_ap, 1), **cnt,
                      "per_class": {c: {**v, **dict(zip(("P", "R", "F1"), map(lambda x: round(x, 1), prf(**v)))),
                                        "AP50": round(100 * aps.get(c, float("nan")), 1)}
                                    for c, v in cls_cnt.items()}},
        "recall_by_zone": {z: {**v, "recall": round(100 * v["tp"] / v["gt"], 1) if v["gt"] else None}
                           for z, v in zone_cnt.items()},
        "miss_analysis": {k: dict(v) for k, v in miss_reason.items()},
        "distance": {"overall": dist_stats(dist_rows),
                     "overall_excluding_clipped": dist_stats([r for r in dist_rows if not r["clipped"]]),
                     "by_zone": {z: dist_stats(by_zone[z]) for z in ZONES if by_zone.get(z)},
                     "by_class": {c: dist_stats(v) for c, v in by_cls.items()},
                     "by_class_zone": {c: dist_stats(v) for c, v in sorted(by_cls_zone.items())}},
    }

    name = a.name or os.path.splitext(os.path.basename(a.model or "default"))[0]
    od = os.path.join(a.out, name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(od, "distance_samples.csv"), "w") as f:
        f.write("class,zone,gt_m,est_m,clipped,view\n")
        for r in dist_rows:
            f.write(f"{r['cls']},{r['zone']},{r['gt']},{r['est']},{r['clipped']},{r['view']}\n")

    d = out["detection"]
    print(f"\n== {name}: {len(imgs)} images ==")
    print(f"P {d['precision']}  R {d['recall']}  F1 {d['f1']}  mAP@0.5 {d['mAP50']}   "
          f"(TP {cnt['tp']} FP {cnt['fp']} FN {cnt['fn']})")
    print("Recall by zone:", {z: v["recall"] for z, v in out["recall_by_zone"].items()})
    print("Distance by zone:")
    for z, s in out["distance"]["by_zone"].items():
        print(f"  {z:8s} {s}")
    print(f"Saved {od}/metrics.json")


if __name__ == "__main__":
    main()
