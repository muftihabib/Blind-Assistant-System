import os

import numpy as np
import torch
from ultralytics import YOLO

from config import DEVICE, IMGSZ, MODEL_PATH, TORCH_THREADS


def load_model(path=None):
    """Load YOLO weights from `path` (default MODEL_PATH); download if missing."""
    path = path or MODEL_PATH
    name = os.path.basename(path)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print(f"[ModelLoader] {path} missing - downloading {name} ...")
        model = YOLO(name)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
    else:
        model = YOLO(path)
    model.to(DEVICE)
    if DEVICE == "cpu":
        torch.set_num_threads(TORCH_THREADS or max(1, (os.cpu_count() or 2) // 2))
    # Warm-up so the first real frame is not slowed by lazy initialisation.
    model.predict(np.zeros((IMGSZ // 2, IMGSZ, 3), np.uint8), imgsz=IMGSZ, verbose=False)
    print(f"[ModelLoader] {path} loaded on {DEVICE} ({torch.get_num_threads()} threads)")
    return model
