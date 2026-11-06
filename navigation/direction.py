def get_direction(position):
    mapping = {
        "LEFT":   "Move right to avoid obstacle on your left",
        "CENTER": "Stop! Obstacle directly ahead",
        "RIGHT":  "Move left to avoid obstacle on your right",
    }
    return mapping.get(position, "Obstacle detected nearby")
