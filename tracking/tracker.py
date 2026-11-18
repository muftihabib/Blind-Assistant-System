"""
Lightweight centroid tracker (§III-F).

Each detection is associated with an existing track of the same class by the
nearest Euclidean centroid distance (greedy, globally closest pairs first),
within TRACK_MAX_MATCH_FRAC * frame width. Unmatched detections start new
tracks; tracks unseen for more than TRACK_MAX_MISSING frames are dropped.

For every detection the tracker sets:
    track_id   persistent integer id
    distance   moving average of distance_raw over the last SMOOTHING_WINDOW
               frames of this track (§III-E.3)
"""

from collections import deque

from config import SMOOTHING_WINDOW, TRACK_MAX_MATCH_FRAC, TRACK_MAX_MISSING


class _Track:
    __slots__ = ("tid", "cls", "centroid", "missing", "dist_hist")

    def __init__(self, tid, cls, centroid, window):
        self.tid = tid
        self.cls = cls
        self.centroid = centroid
        self.missing = 0
        self.dist_hist = deque(maxlen=window)


def _centroid(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class Tracker:
    def __init__(self, max_missing=TRACK_MAX_MISSING, window=SMOOTHING_WINDOW, enabled=True):
        self.max_missing = max_missing
        self.window = window
        self.enabled = enabled
        self.tracks = {}
        self.next_id = 0

    def reset(self):
        self.tracks.clear()

    @property
    def active_ids(self):
        return set(self.tracks.keys())

    def update(self, detections, frame_width):
        if not self.enabled:
            # Ablation mode: no identity across frames, no smoothing.
            for d in detections:
                d["track_id"] = self.next_id
                self.next_id += 1
                d["distance"] = d["distance_raw"]
            return detections

        max_jump = TRACK_MAX_MATCH_FRAC * frame_width
        cents = [_centroid(d["bbox"]) for d in detections]

        pairs = []
        for tid, t in self.tracks.items():
            for i, d in enumerate(detections):
                if d["class"] != t.cls:
                    continue
                dx, dy = cents[i][0] - t.centroid[0], cents[i][1] - t.centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= max_jump:
                    pairs.append((dist, tid, i))
        pairs.sort()

        used_t, used_d = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid); used_d.add(i)
            t = self.tracks[tid]
            t.centroid = cents[i]
            t.missing = 0
            detections[i]["track_id"] = tid

        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]

        for i, d in enumerate(detections):
            if i in used_d:
                continue
            t = _Track(self.next_id, d["class"], cents[i], self.window)
            self.tracks[t.tid] = t
            d["track_id"] = t.tid
            self.next_id += 1

        for d in detections:
            t = self.tracks[d["track_id"]]
            t.dist_hist.append(d["distance_raw"])
            d["distance"] = round(sum(t.dist_hist) / len(t.dist_hist), 3)
        return detections
