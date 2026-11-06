def filter_detections(detections, min_confidence=0.50):
    return [d for d in detections if d["confidence"] >= min_confidence]

def sort_by_size(detections):
    return sorted(detections, key=lambda d: d["pixel_width"], reverse=True)

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)
