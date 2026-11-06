import cv2


class Stream:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def read(self):
        ret, frame = self.camera.read()

        if not ret:
            return None

        return frame

    def release(self):
        self.camera.release()