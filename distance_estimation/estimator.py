from config import (FOCAL_LENGTH, KNOWN_WIDTHS,
                    DISTANCE_DANGER, DISTANCE_MODERATE, DISTANCE_GOOD)

class Estimator:
    def __init__(self):
        self.focal = FOCAL_LENGTH

    def estimate(self, detections, frame_width):
        """Add distance + danger_level + position + nav_direction to each detection."""
        for d in detections:
            dist = self._calc_distance(d["known_width"], d["bbox_width"])
            d["distance"]      = dist
            d["danger_level"]  = self._danger_level(dist)
            d["position"]      = self._position(d["bbox"], frame_width)
            d["nav_direction"] = self._nav_direction(d["bbox"], frame_width, dist)
        return detections

    def _calc_distance(self, real_width_cm, bbox_width_px):
        if bbox_width_px <= 0:
            return 999.0
        dist_cm = (real_width_cm * self.focal) / bbox_width_px
        return round(dist_cm / 100, 2)   # convert cm → metres

    def _danger_level(self, dist_m):
        """
        DANGER   (RED)    → 0 – 1.2 m  : stop / turn immediately
        MODERATE (ORANGE) → 1.2 – 3.0 m: navigate around
        CAUTION  (YELLOW) → 3.0 – 5.0 m: heads-up, slow down
        SAFE     (GREEN)  → > 5.0 m    : continue
        """
        if dist_m <= DISTANCE_DANGER:
            return "DANGER"
        elif dist_m <= DISTANCE_MODERATE:
            return "MODERATE"
        elif dist_m <= DISTANCE_GOOD:
            return "CAUTION"
        else:
            return "SAFE"

    def _position(self, bbox, frame_width):
        """Divide screen into LEFT / CENTER / RIGHT thirds."""
        cx = (bbox[0] + bbox[2]) / 2
        third = frame_width / 3
        if cx < third:
            return "LEFT"
        elif cx < third * 2:
            return "CENTER"
        else:
            return "RIGHT"

    def _nav_direction(self, bbox, frame_width, dist_m):
        """
        Return a navigation instruction (move left / move right) for every object.
        Provides direction regardless of distance so voice always has a suggestion.
        """
        cx    = (bbox[0] + bbox[2]) / 2
        w     = frame_width
        third = w / 3

        # Object on the LEFT  → move right to avoid
        if cx < third:
            return "move right"
        # Object on the RIGHT → move left to avoid
        elif cx > third * 2:
            return "move left"
        else:
            # Object is in the CENTRE
            # Suggest the side with more room (left half → go right, right half → go left)
            if cx < w / 2:
                return "move right"
            else:
                return "move left"
