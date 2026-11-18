"""
Offline speech output with latency logging.

Backends
    pyttsx3 (default, §III-H) — external event loop driven from a worker thread,
              so an emergency alert can interrupt the current utterance.
    sapi    — direct Windows SAPI via win32com (async + purge), fallback.

Queueing: an alert is accepted only if it can START NOW — the speaker is idle,
or the alert is an emergency and the current utterance is not. Otherwise
offer() returns False, the AlertManager does not mark it as announced, and it is
re-proposed on a later frame. No backlog can build up and no accepted alert is
ever silently dropped. (fifo=True is the naive baseline for the ablation: every
alert is queued, nothing is pre-empted.)

Timing caveat: t_audio_start / t_audio_end are the SAPI stream start / end
events. They include the engine's leading/trailing silence and audio-device
buffering (measured here: ~1.2 s for the one-word utterance "Stop"), so they
are an upper bound on audible speech, not an acoustic measurement.

Every alert is logged to ALERT_LOG_FILE with:
    t_capture -> t_queued -> t_audio_start -> t_audio_end
    time_to_first_audio_ms  = t_audio_start - t_capture
    time_to_end_ms          = t_audio_end   - t_capture
    utterance_ms            = t_audio_end   - t_audio_start
    completed               = False if the utterance was interrupted
"""

import csv
import itertools
import os
import threading
from collections import deque
import time

from config import (ALERT_LOG_FILE, TTS_BACKEND, VOICE_RATE, VOICE_VOLUME)

_LOG_FIELDS = ["alert_id", "wall_time", "track_id", "class", "level", "distance_m",
               "emergency", "reason", "text", "chars",
               "t_capture", "t_queued", "t_audio_start", "t_audio_end",
               "capture_to_queue_ms", "time_to_first_audio_ms", "time_to_end_ms",
               "utterance_ms", "completed"]


class _SystemMsg:
    """Non-alert utterance (startup, voice on/off) — not logged."""
    emergency = False

    def __init__(self, text):
        self.text = text


class Speaker:
    def __init__(self, backend=TTS_BACKEND, log_file=ALERT_LOG_FILE, fifo=False):
        self.enabled = True
        self.backend = backend
        self.fifo = fifo                 # ablation: naive unbounded FIFO, no pre-emption
        self._lock = threading.Lock()
        self._queue = deque()
        self._current = None
        self._t_start = None
        self._ready = threading.Event()
        self._stop = False
        self._seq = itertools.count(1)
        self._log_file = log_file
        self._init_log()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        self.offer(_SystemMsg("Blind assistance system ready."))

    # ── public API (main thread) ─────────────────────────────────────────────
    @property
    def busy(self):
        with self._lock:
            return self._current is not None or bool(self._queue)

    @property
    def backlog(self):
        with self._lock:
            return len(self._queue)

    def offer(self, alert):
        """Return True if the alert was accepted for speaking."""
        if not self.enabled and not isinstance(alert, _SystemMsg):
            return False
        with self._lock:
            alert.t_queued = time.perf_counter()
            if self.fifo:
                self._queue.append(alert)
                return True
            cur = self._current
            idle = cur is None and not self._queue
            can_preempt = (alert.emergency
                           and not (cur is not None and getattr(cur, "emergency", False))
                           and not any(getattr(q, "emergency", False) for q in self._queue))
            if idle or can_preempt:
                self._queue.clear()
                self._queue.append(alert)
                return True
        return False

    def say(self, text):
        self.offer(_SystemMsg(text))

    def toggle(self):
        self.enabled = not self.enabled
        state = "ON" if self.enabled else "OFF"
        print(f"[Speaker] Voice {state}")
        msg = _SystemMsg(f"Voice {state}")
        msg.t_queued = time.perf_counter()
        with self._lock:
            self._queue.clear()
            self._queue.append(msg)

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)

    # ── worker thread ────────────────────────────────────────────────────────
    def _run(self):
        try:
            if self.backend == "pyttsx3":
                self._run_pyttsx3()
                return
        except Exception as e:
            print(f"[Speaker] pyttsx3 unavailable ({e}); falling back to SAPI")
        self.backend = "sapi"
        self._run_sapi()

    def _take(self, speaking):
        """Pop the next item if it may start now. Returns (item, interrupt)."""
        with self._lock:
            if not self._queue:
                return None, False
            p = self._queue[0]
            cur = self._current
            # An emergency interrupts ordinary speech, but never another emergency.
            cur_is_emergency = cur is not None and getattr(cur, "emergency", False)
            may_interrupt = not self.fifo and (
                isinstance(p, _SystemMsg) or (p.emergency and not cur_is_emergency))
            if speaking and not may_interrupt:
                return None, False
            self._queue.popleft()
            return p, speaking

    def _started(self, name=None):
        cur = self._current
        if cur is not None and (name is None or name == cur._uname):
            self._t_start = time.perf_counter()

    def _finished(self, completed, name=None):
        t_end = time.perf_counter()
        with self._lock:
            # Ignore late events of an utterance that was already interrupted.
            if self._current is None or (name is not None and name != self._current._uname):
                return
            item, self._current = self._current, None
        if item is not None and not isinstance(item, _SystemMsg):
            self._log(item, self._t_start or t_end, t_end, completed)
        self._t_start = None

    def _run_pyttsx3(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", VOICE_RATE)
        engine.setProperty("volume", VOICE_VOLUME)
        engine.connect("started-utterance", lambda name: self._started(name))
        engine.connect("finished-utterance", lambda name, completed: self._finished(completed, name))
        engine.startLoop(False)
        print("[Speaker] pyttsx3 (offline) ready")
        self._ready.set()
        try:
            while not self._stop:
                speaking = self._current is not None
                item, interrupt = self._take(speaking)
                if item is not None:
                    if interrupt:
                        engine.stop()
                        self._finished(False)   # close it now; its late event is ignored
                    item._uname = f"u{next(self._seq)}"
                    with self._lock:
                        self._current = item
                    if not isinstance(item, _SystemMsg):
                        print(f"[Speaker] >>> {item.text}")
                    engine.say(item.text, item._uname)
                engine.iterate()
                time.sleep(0.005)
        finally:
            engine.endLoop()

    def _run_sapi(self):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        sapi = win32com.client.Dispatch("SAPI.SpVoice")
        sapi.Volume = int(VOICE_VOLUME * 100)
        sapi.Rate = max(-10, min(10, round((VOICE_RATE - 150) / 10)))
        ASYNC, PURGE = 1, 2
        print("[Speaker] SAPI ready")
        self._ready.set()
        started = False
        while not self._stop:
            speaking = self._current is not None
            item, interrupt = self._take(speaking)
            if item is not None:
                if interrupt:
                    self._finished(False)
                item._uname = None
                with self._lock:
                    self._current = item
                if not isinstance(item, _SystemMsg):
                    print(f"[Speaker] >>> {item.text}")
                sapi.Speak(item.text, ASYNC | PURGE)
                started = False
            if self._current is not None:
                state = sapi.Status.RunningState   # 1 = done, 2 = speaking
                if state == 2 and not started:
                    started = True
                    self._started()
                elif state == 1 and sapi.WaitUntilDone(0):
                    self._finished(True)
            time.sleep(0.005)

    # ── logging ──────────────────────────────────────────────────────────────
    def _init_log(self):
        os.makedirs(os.path.dirname(self._log_file) or ".", exist_ok=True)
        if not os.path.exists(self._log_file):
            with open(self._log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_LOG_FIELDS)

    def _log(self, a, t_start, t_end, completed):
        ms = lambda x: round(x * 1000, 1)
        row = [a.alert_id, time.strftime("%Y-%m-%d %H:%M:%S"), a.track_id, a.cls, a.level,
               a.distance, a.emergency, a.reason, a.text, len(a.text),
               round(a.t_capture, 4), round(a.t_queued, 4), round(t_start, 4), round(t_end, 4),
               ms(a.t_queued - a.t_capture), ms(t_start - a.t_capture), ms(t_end - a.t_capture),
               ms(t_end - t_start), completed]
        try:
            with open(self._log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            print(f"[Speaker] log error: {e}")
