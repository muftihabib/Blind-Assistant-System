"""Unit tests for the decision logic.  Run:  python -m pytest -q tests"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distance_estimation.estimator import Estimator, danger_level, select_width
from navigation.guidance import compute_guidance
from tracking.tracker import Tracker
from voice.alerts import AlertManager

W, H = 1280, 720


def det(cls, x1, x2, dist=None, y1=200, y2=600, tid=0):
    d = {"class": cls, "bbox": (x1, y1, x2, y2), "bbox_width": x2 - x1,
         "bbox_height": y2 - y1, "track_id": tid, "position": "CENTER", "clipped": False}
    if dist is not None:
        d["distance"] = dist
        d["danger_level"] = danger_level(dist)
    return d


# ── danger levels (Table III) ────────────────────────────────────────────────
def test_levels():
    assert danger_level(1.0) == "DANGER"
    assert danger_level(1.01) == "WARNING"
    assert danger_level(2.5) == "WARNING"
    assert danger_level(4.0) == "CAUTION"
    assert danger_level(4.01) == "SAFE"


# ── width priors (Table IV) ─────────────────────────────────────────────────
def test_side_profile_switch():
    assert select_width("car", 300, 250) == (180, "front")    # AR 1.2
    assert select_width("car", 450, 250) == (420, "side")     # AR 1.8
    assert select_width("bicycle", 200, 200) == (70, "front")
    assert select_width("bicycle", 260, 200) == (175, "side")
    assert select_width("person", 600, 200)[1] == "front"     # person never switches


# ── pinhole + clipping ───────────────────────────────────────────────────────
def test_pinhole_and_clipping():
    e = Estimator(W, H)
    f = e.focal
    d = det("person", 500, 500 + 45 * f / 200)                 # person at 2.00 m
    e.estimate([d], W, H)
    assert abs(d["distance_raw"] - 2.0) < 0.02 and not d["clipped"]

    near = det("person", 0, 900)                              # touches left edge
    e.estimate([near], W, H)
    near["distance"] = near["distance_raw"]
    e.finalize([near], W)
    assert near["clipped"] and near["danger_level"] == "DANGER"   # 900 px >= 50% of frame

    both = det("car", 0, W - 1)
    e.estimate([both], W, H)
    both["distance"] = both["distance_raw"]
    e.finalize([both], W)
    assert both["clipped_both"] and both["danger_level"] == "DANGER"


def test_occlusion_flag():
    e = Estimator(W, H)
    back = det("person", 500, 700, y1=200, y2=500)
    front = det("chair", 520, 720, y1=250, y2=650)
    e.estimate([back, front], W, H)
    assert back["occluded"] and not front["occluded"]


# ── path guidance (§III-G) ───────────────────────────────────────────────────
def test_guidance_rules():
    centre = det("person", 560, 720, 1.5)
    assert compute_guidance([det("person", 560, 720, 3.0)], W)["action"] == "CLEAR"
    assert compute_guidance([centre, det("chair", 950, 1100, 2.0)], W)["action"] == "MOVE_LEFT"
    assert compute_guidance([centre, det("chair", 100, 300, 2.0)], W)["action"] == "MOVE_RIGHT"
    assert compute_guidance([centre, det("chair", 100, 300, 2.0),
                             det("car", 950, 1200, 2.8)], W)["action"] == "STOP"
    # both clear -> side with more clearance
    assert compute_guidance([centre, det("chair", 100, 300, 3.5)], W)["action"] == "MOVE_RIGHT"
    # no other obstacles, hazard left of centre -> move right
    assert compute_guidance([det("person", 480, 600, 1.2)], W)["action"] == "MOVE_RIGHT"
    # a wide object spanning LEFT+CENTER blocks the left side
    g = compute_guidance([det("car", 100, 700, 2.0)], W)
    assert g["action"] == "MOVE_RIGHT" and not g["clear"]["LEFT"]


# ── alerts: cooldown and override (§III-H) ───────────────────────────────────
def test_cooldown_and_override():
    m = AlertManager(cooldown=3.0, approach_delta=0.5)
    g = {"hazard": None, "message": None}
    d = det("person", 560, 720, 2.0, tid=7)
    a = m.select([d], g, 0.0, now=0.0)
    assert a and a.reason == "new"
    m.commit(a, [d])
    m._last[7] = (0.0, 2.0, "WARNING")

    assert m.select([d], g, 0.0, now=1.0) is None                     # in cooldown
    d["distance"] = 1.45                                               # 0.55 m closer
    a = m.select([d], g, 0.0, now=1.2)
    assert a and a.emergency and a.reason == "approach"

    m._last[7] = (1.2, 1.45, "WARNING")
    d["distance"], d["danger_level"] = 0.95, "DANGER"
    a = m.select([d], g, 0.0, now=1.4)
    assert a and a.reason in ("enter_danger", "approach") and a.emergency

    m._last[7] = (1.4, 0.95, "DANGER")
    assert m.select([d], g, 0.0, now=2.0) is None
    assert m.select([d], g, 0.0, now=4.5).reason == "cooldown"


def test_priority_order():
    m = AlertManager()
    g = {"hazard": None, "message": None}
    chair = det("chair", 100, 300, 0.8, tid=1)
    person = det("person", 900, 1000, 0.9, tid=2)
    far = det("car", 600, 700, 3.0, tid=3)
    assert m.select([far, chair, person], g, 0.0, now=0).track_id == 2   # person before chair at DANGER


def test_guidance_appended_to_hazard_alert():
    m = AlertManager()
    centre = det("person", 560, 720, 0.8, tid=1)
    g = compute_guidance([centre], W)
    a = m.select([centre], g, 0.0, now=0)
    assert a.text.endswith(g["message"])


# ── tracking + smoothing ─────────────────────────────────────────────────────
def test_tracker_ids_and_smoothing():
    t = Tracker(window=5)
    ids = []
    for i, dist in enumerate([2.0, 2.2, 1.8, 2.0, 2.0, 3.0]):
        d = det("person", 500 + 5 * i, 640 + 5 * i)
        d["distance_raw"] = dist
        t.update([d], W)
        ids.append(d["track_id"])
    assert len(set(ids)) == 1
    assert abs(d["distance"] - (2.2 + 1.8 + 2.0 + 2.0 + 3.0) / 5) < 1e-6

    far_jump = det("person", 1100, 1240)
    far_jump["distance_raw"] = 2.0
    t.update([far_jump], W)
    assert far_jump["track_id"] != ids[0]
