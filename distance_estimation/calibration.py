from config import FOCAL_LENGTH

def calibrate(known_width_cm, actual_distance_cm, measured_pixel_width):
    focal = (measured_pixel_width * actual_distance_cm) / known_width_cm
    print(f"Calibrated Focal Length: {focal:.2f}")
    return focal
