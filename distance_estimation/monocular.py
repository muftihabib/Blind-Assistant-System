from config import FOCAL_LENGTH

def calculate_distance(known_width_cm, pixel_width):
    if not known_width_cm or pixel_width <= 0:
        return None
    distance_cm = (known_width_cm * FOCAL_LENGTH) / pixel_width
    return round(distance_cm / 100, 2)
