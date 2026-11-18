import threading
import time

import cv2

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH


class Webcam:
    """
    Live camera (background thread keeps only the newest frame) or a video
    file (read sequentially so experiments are repeatable).

    read() -> (ok, frame, t_capture)   with t_capture from time.perf_counter().
    """

    def __init__(self, source=CAMERA_INDEX):
        self.is_file = isinstance(source, str) and not source.isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.t_capture = None, None
        self.lock = threading.Lock()
        self.running = False
        self.ended = False

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera / video source.")
        self.running = True
        if not self.is_file:
            threading.Thread(target=self._reader, daemon=True).start()
            time.sleep(0.5)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
                print(f"[Webcam] WARNING: camera delivers {w}x{h}, "
                      f"config asks {FRAME_WIDTH}x{FRAME_HEIGHT}")
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame, self.t_capture = frame, time.perf_counter()
            else:
                time.sleep(0.01)

    def read(self):
        if self.is_file:
            ret, frame = self.cap.read()
            if not ret:
                self.ended = True
                return False, None, None
            return True, frame, time.perf_counter()
        with self.lock:
            if self.frame is None:
                return False, None, None
            return True, self.frame.copy(), self.t_capture

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()
