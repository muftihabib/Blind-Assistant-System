import cv2
import time
import os

def save_screenshot(frame, folder="screenshots"):
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"capture_{int(time.time())}.jpg")
    cv2.imwrite(fname, frame)
    print(f"Screenshot saved: {fname}")
    return fname
