# navigation/guidance.py

from navigation.direction import get_direction
from navigation.obstacle  import get_closest, is_blocking

def generate_guidance(detections):
    """Return a guidance string based on detections"""
    closest = get_closest(detections)
    if not closest:
        return "Path is clear."
    if is_blocking(closest):
        return get_direction(closest.get('position','CENTER'))
    return f"{closest['class']} ahead at {closest['distance_m']} metres."
