import os
from ultralytics import YOLO
from config import MODEL_PATH, DEVICE

def load_model():
    # Determine which model to download based on MODEL_PATH
    model_name = os.path.basename(MODEL_PATH)   # e.g. "yolov8s.pt"

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        print(f"[ModelLoader] {MODEL_PATH} missing/corrupt – downloading {model_name} …")
        model = YOLO(model_name)   # downloads from ultralytics hub
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
    else:
        model = YOLO(MODEL_PATH)

    model.to(DEVICE)
    print(f"[ModelLoader] Model loaded on: {DEVICE}")
    return model
