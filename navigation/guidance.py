from navigation.direction import get_direction
from navigation.obstacle  import get_closest, is_blocking

def generate_guidance(detections):
    closest = get_closest(detections)
    if not closest:
        return "Path is clear."
    if is_blocking(closest):
        return get_direction(closest.get("position", "CENTER"))
    cls_name = closest["class"]
    dist     = closest["distance_m"]
    return f"{cls_name} ahead at {dist} metres."
