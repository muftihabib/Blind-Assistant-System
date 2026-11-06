from config import CONFIDENCE, IOU_THRESHOLD

def run_inference(model, frame):
    try:
        results = model(frame, conf=CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
        return results
    except Exception as e:
        print(f"Inference error: {e}")
        return []
