"""
Alert management (§III-H).

Priority  : (danger level, high-priority class first, distance) — most urgent first.
Cooldown  : a tracked object (track_id) is announced at most once per
            ALERT_COOLDOWN seconds ...
Override  : ... unless, since its last announcement, it has come
            >= APPROACH_DELTA metres closer, or it has newly entered DANGER.
            An override alert is marked `emergency` and pre-empts current speech.
Guidance  : when the selected object is the centre hazard, the steering cue
            (Move left / Move right / Path blocked. Stop.) is appended.

AlertManager.select() only proposes an alert; commit() records it once the
speaker has accepted it, so an alert dropped because the speaker was busy is
not counted as announced.
"""

import itertools
import time
from dataclasses import dataclass, field

from config import (ALERT_COOLDOWN, ANNOUNCE_LEVELS, APPROACH_DELTA,
                    HIGH_PRIORITY)
from distance_estimation.estimator import LEVEL_RANK

_ids = itertools.count(1)


@dataclass
class Alert:
    text: str
    track_id: int
    cls: str
    level: str
    distance: float
    emergency: bool
    reason: str                     # "new" | "cooldown" | "approach" | "enter_danger" | "no_cooldown"
    t_capture: float                # frame capture time (perf_counter)
    t_created: float = field(default_factory=time.perf_counter)
    alert_id: int = field(default_factory=lambda: next(_ids))


def _pos_phrase(pos):
    return {"LEFT": "on left", "RIGHT": "on right"}.get(pos, "ahead")


def _dist_phrase(d):
    dist = d["distance"]
    if d.get("clipped"):
        # Box truncated by the frame edge: distance is only an upper bound.
        return "very close" if dist <= 1.0 else f"within {dist:.0f} meters"
    if dist < 0.5:
        return "very close"
    if dist < 1.0:
        return f"{int(round(dist * 100, -1))} centimeters"
    if dist < 2.0:
        return f"{dist:.1f} meters"
    return f"{dist:.0f} meters"


def format_alert(d, guidance_msg=None):
    """
    Short, minimally punctuated phrases keep time-to-end-of-utterance low
    (SAPI pauses ~0.7 s at each '.' / '!' and ~0.4 s at each ','), e.g.
        "Danger person ahead very close, move left"
        "Danger chair on right 80 centimeters"
        "Warning car ahead 1.8 meters, stop, path blocked"
        "Caution bicycle on left 3 meters"
    """
    cls, level = d["class"], d["danger_level"]
    text = f"{level.capitalize()} {cls} {_pos_phrase(d['position'])} {_dist_phrase(d)}"
    if guidance_msg and level in ("DANGER", "WARNING"):
        text += f", {guidance_msg}"
    elif level == "DANGER" and d["position"] == "CENTER":
        text += ", stop"
    return text


class AlertManager:
    def __init__(self, cooldown=ALERT_COOLDOWN, approach_delta=APPROACH_DELTA, use_cooldown=True):
        self.cooldown = cooldown
        self.approach_delta = approach_delta
        self.use_cooldown = use_cooldown
        self._last = {}     # track_id -> (time, distance, level)

    @staticmethod
    def priority_key(d):
        return (LEVEL_RANK[d["danger_level"]],
                0 if d["class"] in HIGH_PRIORITY else 1,
                d["distance"])

    def _eligibility(self, d, now):
        """Return (eligible, emergency, reason)."""
        if not self.use_cooldown:
            return True, False, "no_cooldown"
        last = self._last.get(d["track_id"])
        if last is None:
            return True, d["danger_level"] == "DANGER", "new"
        t, dist, level = last
        if d["danger_level"] == "DANGER" and level != "DANGER":
            return True, True, "enter_danger"
        if dist - d["distance"] >= self.approach_delta:
            return True, True, "approach"
        if now - t >= self.cooldown:
            return True, False, "cooldown"
        return False, False, ""

    def select(self, detections, guidance, t_capture, now=None):
        now = time.perf_counter() if now is None else now
        cands = [d for d in detections if d["danger_level"] in ANNOUNCE_LEVELS]
        cands.sort(key=self.priority_key)
        hazard = guidance.get("hazard") if guidance else None

        for d in cands:
            ok, emergency, reason = self._eligibility(d, now)
            if not ok:
                continue
            g = guidance["message"] if (hazard is not None and d is hazard) else None
            return Alert(format_alert(d, g), d["track_id"], d["class"], d["danger_level"],
                         d["distance"], emergency, reason, t_capture)
        return None

    def commit(self, alert, detections):
        now = time.perf_counter()
        for d in detections:
            if d["track_id"] == alert.track_id:
                self._last[d["track_id"]] = (now, d["distance"], d["danger_level"])
                break

    def prune(self, active_track_ids):
        for tid in list(self._last):
            if tid not in active_track_ids:
                del self._last[tid]
