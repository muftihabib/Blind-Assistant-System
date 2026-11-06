def position_in_frame(bbox, frame_width):
    cx    = (bbox[0] + bbox[2]) // 2
    third = frame_width // 3
    if cx < third:
        return "LEFT"
    elif cx < 2 * third:
        return "CENTER"
    else:
        return "RIGHT"
