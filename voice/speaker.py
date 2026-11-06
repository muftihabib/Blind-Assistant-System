import win32com.client
import threading
import time
from config import VOICE_RATE, VOICE_VOLUME

# SAPI async flag — non-blocking
_ASYNC = 1

# How often to speak (seconds) — speech generated fresh each time
SPEECH_INTERVAL = 2.5

# Sort order (lower = more urgent)
_LEVEL_ORDER = {"DANGER": 0, "MODERATE": 1, "CAUTION": 2, "SAFE": 3}


class Speaker:
    """
    Every SPEECH_INTERVAL seconds, reads the LATEST detections snapshot
    and speaks the most urgent object with its CURRENT distance.

    This ensures distance spoken == distance shown on screen.
    Voice starts automatically — no key press needed.
    """

    def __init__(self):
        self.enabled           = True
        self._latest           = []       # always holds the most recent detections
        self._latest_lock      = threading.Lock()
        self._sapi             = None

        # Background thread: init SAPI + speak loop
        threading.Thread(target=self._run, daemon=True).start()
        print("[Speaker] Voice system ready – speaking automatically")

    # ── SAPI init + speak loop (single background thread) ───────────────────
    def _run(self):
        """Init SAPI, announce startup, then speak every SPEECH_INTERVAL seconds."""
        try:
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
            self._sapi.Rate   = self._rate_from_config()
            self._sapi.Volume = int(VOICE_VOLUME * 100)
        except Exception as e:
            print(f"[Speaker] SAPI init error: {e}")
            return

        # Startup announcement
        self._say_blocking("Blind assistance system ready.")

        # Speak loop — every SPEECH_INTERVAL seconds speak from LATEST snapshot
        while True:
            time.sleep(SPEECH_INTERVAL)

            if not self.enabled:
                continue

            with self._latest_lock:
                detections = list(self._latest)   # copy latest snapshot

            if not detections:
                continue

            msg = self._build_message(detections)
            if msg:
                print(f"[Speaker] >>> {msg}")
                self._say_blocking(msg)

    def _say_blocking(self, text):
        """Speak text and WAIT for it to finish before returning."""
        try:
            self._sapi.Speak(text, 0)   # 0 = synchronous (blocking until done)
        except Exception as e:
            print(f"[Speaker] Speak error: {e}")

    # ── Called by main loop every detection cycle ────────────────────────────
    def speak(self, detections):
        """
        Store the latest detections snapshot.
        The background thread reads this and speaks from it every 2.5 seconds.
        Distance spoken is ALWAYS current — no stale data.
        """
        with self._latest_lock:
            self._latest = detections   # replace with freshest data

    # ── Build spoken message from detections ─────────────────────────────────
    def _build_message(self, detections):
        """Announce ALL detected objects, sorted by urgency (most dangerous first)."""

        if not detections:
            return None

        # Sort: most dangerous + closest first
        ordered = sorted(
            detections,
            key=lambda d: (
                _LEVEL_ORDER.get(d.get("danger_level", "SAFE"), 3),
                d.get("distance", 999.0)
            )
        )

        # Limits per danger level to keep message length reasonable
        # DANGER: up to 4, MODERATE: up to 3, CAUTION: up to 2, SAFE: up to 1
        _LIMITS = {"DANGER": 4, "MODERATE": 3, "CAUTION": 2, "SAFE": 1}
        _level_counts = {"DANGER": 0, "MODERATE": 0, "CAUTION": 0, "SAFE": 0}

        parts = []
        for det in ordered:
            level = det.get("danger_level", "SAFE")
            limit = _LIMITS.get(level, 1)
            if _level_counts[level] >= limit:
                continue
            _level_counts[level] += 1

            phrase = self._format_single(det)
            if phrase:
                parts.append(phrase)

        if not parts:
            return None

        return " ... ".join(parts)

    def _format_single(self, det):
        """Format a single detection into a spoken phrase."""
        cls   = det.get("class",        "object")
        dist  = det.get("distance",      999.0)
        level = det.get("danger_level", "SAFE")
        pos   = det.get("position",     "CENTER")

        # Position phrase
        if pos == "LEFT":
            pos_phrase = "on your left"
        elif pos == "RIGHT":
            pos_phrase = "on your right"
        else:
            pos_phrase = "directly ahead"

        # Distance phrase — rounded to avoid false precision
        if dist < 0.5:
            dist_phrase = "very close"
        elif dist < 1.0:
            dist_phrase = f"{int(dist * 100)} centimetres"
        elif dist < 2.0:
            dist_phrase = f"{dist:.1f} metres"
        else:
            dist_phrase = f"{round(dist, 0):.0f} metres"

        # Full message per danger level
        if level == "DANGER":
            return f"Danger! {cls} {pos_phrase}, {dist_phrase}. Stop!"
        elif level == "MODERATE":
            return f"Warning. {cls} {pos_phrase}, {dist_phrase}."
        elif level == "CAUTION":
            return f"Caution, {cls} {pos_phrase}, {dist_phrase}."
        else:
            return f"{cls} {pos_phrase}, {dist_phrase}."

    def _rate_from_config(self):
        rate_map = {120: -3, 130: -2, 140: -1, 150: 0,
                    160: 1,  170: 2,  180: 3,  200: 4}
        closest = min(rate_map.keys(), key=lambda k: abs(k - VOICE_RATE))
        return rate_map[closest]

    def say(self, text):
        """Speak arbitrary text immediately."""
        if self._sapi:
            try:
                self._sapi.Speak(text, _ASYNC)
            except Exception:
                pass

    def toggle(self):
        """V key — mute / unmute."""
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"[Speaker] Voice {state}")
        self.say(f"Voice {state}")
