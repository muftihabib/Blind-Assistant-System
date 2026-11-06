"""
Read frames from a video file.
"""

import cv2


class VideoReader:

    def __init__(self, path):

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

    def read(self):

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):

        self.cap.release()