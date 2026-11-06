import cv2
import numpy as np
import math
import time

C = {
    "DANGER":   (0,  50, 255),    
    "MODERATE": (0, 140, 255),    
    "CAUTION":  (0, 210, 255),   
    "SAFE":     (255, 210,  0),   
    "HUD":      (0, 185, 255),       
    "CYAN":     (255, 220,  0),   
    "WHITE":    (240, 240, 240),
    "DIM":      (80,  80,  80),
    "PANEL":    (15,  15,  20),
}

_START = time.time()


def _glow_rect(img, pt1, pt2, color, thickness=2, glow_layers=3):
    """Draw a glowing rectangle using multiple semi-transparent layers."""
    overlay = img.copy()
    for i in range(glow_layers, 0, -1):
        alpha = 0.15 * i
        thick = thickness + i * 2
        cv2.rectangle(overlay, pt1, pt2, color, thick)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, color, thickness)


def _glow_text(img, text, org, font, scale, color, thick=2):
    """Draw glowing text."""
    overlay = img.copy()
    for i in range(3, 0, -1):
        cv2.putText(overlay, text, org, font, scale, color, thick + i * 2)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def _corner_bracket(img, x1, y1, x2, y2, color, arm=22, thick=2):
    """Draw futuristic corner brackets instead of a full rectangle."""
    pts = [
        # Top-left
        ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
        # Top-right
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        # Bottom-left
        ((x1, y2 - arm), (x1, y2), (x1 + arm, y2)),
        # Bottom-right
        ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),
    ]
    for p in pts:
        cv2.polylines(img, [np.array(p, dtype=np.int32)], False, color, thick, cv2.LINE_AA)


def _draw_grid(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    step = 80
    col = (0, 160, 255)
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), col, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), col, 1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)



def _draw_radar(frame, object_count):
    cx, cy, r = 68, 68, 52
    t = time.time() - _START
    angle = int((t * 60) % 360)

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r + 6, C["HUD"], 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    for ri in [r, r // 2]:
        cv2.circle(frame, (cx, cy), ri, C["HUD"], 1, cv2.LINE_AA)

    cv2.line(frame, (cx - r, cy), (cx + r, cy), C["HUD"], 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), C["HUD"], 1, cv2.LINE_AA)

    # Sweep arc (filled sector)
    sweep = frame.copy()
    cv2.ellipse(sweep, (cx, cy), (r, r), 0, -angle - 40, -angle, C["HUD"], -1)
    cv2.addWeighted(sweep, 0.2, frame, 0.8, 0, frame)

    # Sweep line
    rad = math.radians(-angle)
    ex  = int(cx + r * math.cos(rad))
    ey  = int(cy + r * math.sin(rad))
    cv2.line(frame, (cx, cy), (ex, ey), C["HUD"], 2, cv2.LINE_AA)

    # Centre dot
    cv2.circle(frame, (cx, cy), 4, C["HUD"], -1)

    # Object count inside radar
    label = str(object_count)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, label, (cx - tw // 2, cy + r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["CYAN"], 1, cv2.LINE_AA)


# ── Right sidebar panel ───────────────────────────────────────────────────────
def _draw_sidebar(frame, detections):
    h, w   = frame.shape[:2]
    sw     = 240           # sidebar width
    sx     = w - sw
    margin = 10
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # Dark semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (sx, 0), (w, h), C["PANEL"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Left border glow
    cv2.line(frame, (sx, 0), (sx, h), C["HUD"], 2, cv2.LINE_AA)
    overlay2 = frame.copy()
    cv2.line(overlay2, (sx - 3, 0), (sx - 3, h), C["HUD"], 1)
    cv2.addWeighted(overlay2, 0.3, frame, 0.7, 0, frame)

    # Header
    cv2.putText(frame, "DETECTIONS", (sx + margin, 28),
                font, 0.55, C["HUD"], 1, cv2.LINE_AA)
    cv2.line(frame, (sx + margin, 34), (w - margin, 34), C["HUD"], 1)

    # Sort: most dangerous + closest first
    _ORD = {"DANGER": 0, "MODERATE": 1, "CAUTION": 2, "SAFE": 3}
    ordered = sorted(detections,
                     key=lambda d: (_ORD.get(d.get("danger_level", "SAFE"), 3),
                                    d.get("distance", 999)))

    y = 56
    for i, d in enumerate(ordered[:10]):   # max 10 items
        cls   = d.get("class", "?").upper()
        dist  = d.get("distance", 0)
        level = d.get("danger_level", "SAFE")
        pos   = d.get("position", "CENTER")
        color = C.get(level, C["SAFE"])

        # Row background (alternating)
        if i % 2 == 0:
            row_overlay = frame.copy()
            cv2.rectangle(row_overlay, (sx + 2, y - 14), (w - 2, y + 8),
                          (30, 30, 40), -1)
            cv2.addWeighted(row_overlay, 0.5, frame, 0.5, 0, frame)

        # Coloured level indicator bar
        cv2.rectangle(frame, (sx + margin, y - 11), (sx + margin + 4, y + 5),
                      color, -1)

        # Object name
        cv2.putText(frame, cls, (sx + margin + 10, y),
                    font, 0.48, color, 1, cv2.LINE_AA)

        # Distance
        if dist < 1.0:
            dist_str = f"{int(dist*100)}cm"
        else:
            dist_str = f"{dist:.1f}m"
        cv2.putText(frame, dist_str, (sx + sw - 55, y),
                    font, 0.48, C["WHITE"], 1, cv2.LINE_AA)

        # Position tag
        pos_abbr = {"LEFT": "L", "RIGHT": "R", "CENTER": "C"}.get(pos, pos[0])
        cv2.putText(frame, pos_abbr, (w - margin - 12, y),
                    font, 0.42, C["CYAN"], 1, cv2.LINE_AA)

        y += 24
        if y > h - 40:
            break

    # Footer
    cv2.line(frame, (sx + margin, h - 30), (w - margin, h - 30), C["HUD"], 1)
    t_str = f"SORT: BY DISTANCE"
    cv2.putText(frame, t_str, (sx + margin, h - 12),
                font, 0.38, C["DIM"], 1, cv2.LINE_AA)


# ── Detection boxes ───────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        level  = d.get("danger_level", "SAFE")
        color  = C.get(level, C["SAFE"])
        dist   = d.get("distance", 0)
        cls    = d.get("class", "object").upper()
        pos    = d.get("position", "CENTER")

        # Clamp to frame
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        # ── Glow effect behind box ────────────────────────────────────────
        glow = frame.copy()
        cv2.rectangle(glow, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 8)
        cv2.addWeighted(glow, 0.25, frame, 0.75, 0, frame)

        # ── Corner brackets ───────────────────────────────────────────────
        _corner_bracket(frame, x1, y1, x2, y2, color, arm=20, thick=2)

        # ── Label: "LAPTOP 1.8m" ──────────────────────────────────────────
        if dist < 1.0:
            dist_str = f"{int(dist*100)}cm"
        else:
            dist_str = f"{dist:.1f}m"

        label = f"{cls} {dist_str}"
        scale = 0.65
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        lx = x1
        ly = max(y1 - th - 10, 0)

        # Label background (dark)
        label_bg = frame.copy()
        cv2.rectangle(label_bg, (lx - 2, ly - 2), (lx + tw + 10, ly + th + 6),
                      (10, 10, 15), -1)
        cv2.addWeighted(label_bg, 0.8, frame, 0.2, 0, frame)

        # Label border
        cv2.rectangle(frame, (lx - 2, ly - 2), (lx + tw + 10, ly + th + 6),
                      color, 1)

        # Label text with glow
        _glow_text(frame, label, (lx + 4, ly + th), font, scale, color, thick)

        # ── Navigation arrow inside box ───────────────────────────────────
        nav = d.get("nav_direction")
        if nav and level in ("DANGER", "MODERATE"):
            arrow = "<< MOVE LEFT" if nav == "move left" else "MOVE RIGHT >>"
            (aw, ah), _ = cv2.getTextSize(arrow, font, 0.5, 1)
            ax = x1 + ((x2 - x1) - aw) // 2
            ay = (y1 + y2) // 2 + ah // 2
            cv2.putText(frame, arrow, (ax, ay), font, 0.5, color, 1, cv2.LINE_AA)

    return frame


# ── Top HUD bar ───────────────────────────────────────────────────────────────
def _draw_top_bar(frame, fps, count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (10, 10, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, 44), (w, 44), C["HUD"], 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Left: system title
    cv2.putText(frame, "BLIND ASSIST AI", (130, 28),
                font, 0.65, C["HUD"], 1, cv2.LINE_AA)

    # Centre: FPS
    fps_str = f"FPS  {fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, font, 0.55, 1)
    cv2.putText(frame, fps_str, (w // 2 - tw // 2, 28),
                font, 0.55, C["CYAN"], 1, cv2.LINE_AA)

    # Right: controls hint
    cv2.putText(frame, "V=MUTE   Q=QUIT", (w - 260, 28),
                font, 0.45, C["DIM"], 1, cv2.LINE_AA)


# ── Bottom bar ────────────────────────────────────────────────────────────────
def _draw_bottom_bar(frame, count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (10, 10, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, h - 36), (w, h - 36), C["HUD"], 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"Objects Detected:  {count}"
    _glow_text(frame, label, (130, h - 12), font, 0.6, C["HUD"])


# ── Scan line effect ──────────────────────────────────────────────────────────
def _draw_scanline(frame):
    h, w = frame.shape[:2]
    t = time.time() - _START
    y = int((t * 120) % h)
    overlay = frame.copy()
    cv2.line(overlay, (0, y), (w, y), C["CYAN"], 1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


# ── Main HUD compositor ───────────────────────────────────────────────────────
def draw_hud(frame, fps, count, detections=None):
    detections = detections or []

    _draw_grid(frame)
    _draw_top_bar(frame, fps, count)
    _draw_radar(frame, count)
    _draw_bottom_bar(frame, count)
    _draw_scanline(frame)

    if detections:
        _draw_sidebar(frame, detections)

    # Navigation banner (centre-top) for most urgent DANGER/MODERATE
    _ORD = {"DANGER": 0, "MODERATE": 1, "CAUTION": 2, "SAFE": 3}
    if detections:
        urgent = sorted(detections,
                        key=lambda d: (_ORD.get(d.get("danger_level", "SAFE"), 3),
                                       d.get("distance", 999)))
        for d in urgent:
            nav   = d.get("nav_direction")
            level = d.get("danger_level", "SAFE")
            if nav and level in ("DANGER", "MODERATE"):
                color  = C.get(level, C["HUD"])
                banner = f"<< MOVE LEFT" if nav == "move left" else "MOVE RIGHT >>"
                font   = cv2.FONT_HERSHEY_SIMPLEX
                scale  = 1.3
                (bw, bh), _ = cv2.getTextSize(banner, font, scale, 3)
                bx = (frame.shape[1] - bw) // 2
                by = 90
                # Glow background
                bg = frame.copy()
                cv2.rectangle(bg, (bx - 10, by - bh - 8), (bx + bw + 10, by + 8),
                              (10, 10, 18), -1)
                cv2.addWeighted(bg, 0.7, frame, 0.3, 0, frame)
                _glow_text(frame, banner, (bx, by), font, scale, color, 3)
                break

    return frame
