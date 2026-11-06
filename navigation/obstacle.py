def is_blocking(detection):
    """
    Returns True if the object is directly ahead and close enough to block the path.
    Uses 'distance' (not 'distance_m') to match the Estimator output.
    """
    return (
        detection.get("danger_level") in ("DANGER", "MODERATE") and
        detection.get("position") == "CENTER"
    )


def get_closest(detections):
    """
    Returns the nearest detection from the list.
    Uses 'distance' key (set by Estimator).
    """
    valid = [d for d in detections if d.get("distance") is not None]
    if not valid:
        return None
    return min(valid, key=lambda d: d["distance"])
