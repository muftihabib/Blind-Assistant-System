import cv2, threading, time
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class Webcam:
    def __init__(self):
        self.cap    = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ← key fix: 1-frame buffer
        self.frame  = None
        self.lock   = threading.Lock()
        self.running = False

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera nahi khuli! Index check karo.")
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(0.5)   # warm-up
        return self

    def _reader(self):
        """Background thread – hamesha latest frame rakhta hai"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        time.sleep(0.3)
        self.cap.release()
