"""
Safe-path guidance (§III-G).

Sectors (W = frame width):
    LEFT   = [0,        a*W)
    CENTER = [a*W,      b*W]
    RIGHT  = (b*W,      W]          with (a, b) = SECTOR_SPLITS = (1/3, 2/3)

An obstacle OCCUPIES a sector when its bounding box covers at least
SECTOR_MIN_OVERLAP of that sector's width (so a wide object can occupy
several sectors; a centroid-only rule would miss a car spanning LEFT+CENTER).

Definitions (one consistent pair):
    Hazard ahead : some obstacle occupying CENTER has D <= GUIDANCE_TRIGGER (2.5 m)
    Side Clear   : every obstacle occupying the side has D > CLEAR_DISTANCE (3.0 m)
                   (an empty sector is Clear)
    Side Blocked : not Clear

Decision rules (exhaustive):
    1. No hazard ahead                       -> CLEAR       (no steering cue)
    2. Hazard, LEFT Clear,  RIGHT Blocked    -> MOVE_LEFT
    3. Hazard, LEFT Blocked, RIGHT Clear     -> MOVE_RIGHT
    4. Hazard, both Clear                    -> side with the larger clearance
                                                (nearest obstacle distance; empty = inf);
                                                tie -> away from the hazard's centroid
    5. Hazard, both Blocked                  -> STOP
"""

import math

from config import (CLEAR_DISTANCE, GUIDANCE_TRIGGER, SECTOR_MIN_OVERLAP,
                    SECTOR_SPLITS)

SECTORS = ("LEFT", "CENTER", "RIGHT")
MESSAGES = {            # spoken after the hazard alert; no sentence punctuation (SAPI
    "CLEAR": None,      # inserts ~0.7 s of silence at every '.' or '!')
    "MOVE_LEFT": "move left",
    "MOVE_RIGHT": "move right",
    "STOP": "stop, path blocked",
}


def sector_bounds(frame_width):
    a, b = SECTOR_SPLITS
    return {"LEFT": (0.0, a * frame_width),
            "CENTER": (a * frame_width, b * frame_width),
            "RIGHT": (b * frame_width, float(frame_width))}


def occupied_sectors(bbox, frame_width):
    out = []
    for name, (s0, s1) in sector_bounds(frame_width).items():
        overlap = max(0.0, min(bbox[2], s1) - max(bbox[0], s0))
        if overlap >= SECTOR_MIN_OVERLAP * (s1 - s0):
            out.append(name)
    return out


def compute_guidance(detections, frame_width):
    """Return {"action", "message", "hazard", "clearance", "clear"}."""
    nearest = {s: math.inf for s in SECTORS}
    nearest_det = {s: None for s in SECTORS}
    for d in detections:
        for s in occupied_sectors(d["bbox"], frame_width):
            if d["distance"] < nearest[s]:
                nearest[s] = d["distance"]
                nearest_det[s] = d

    clear = {s: nearest[s] > CLEAR_DISTANCE for s in ("LEFT", "RIGHT")}
    hazard = nearest_det["CENTER"] if nearest["CENTER"] <= GUIDANCE_TRIGGER else None

    if hazard is None:
        action = "CLEAR"
    elif clear["LEFT"] and not clear["RIGHT"]:
        action = "MOVE_LEFT"
    elif clear["RIGHT"] and not clear["LEFT"]:
        action = "MOVE_RIGHT"
    elif clear["LEFT"] and clear["RIGHT"]:
        if nearest["LEFT"] != nearest["RIGHT"]:
            action = "MOVE_LEFT" if nearest["LEFT"] > nearest["RIGHT"] else "MOVE_RIGHT"
        else:
            hx = (hazard["bbox"][0] + hazard["bbox"][2]) / 2
            action = "MOVE_RIGHT" if hx < frame_width / 2 else "MOVE_LEFT"
    else:
        action = "STOP"

    return {"action": action, "message": MESSAGES[action], "hazard": hazard,
            "clearance": nearest, "clear": clear}
