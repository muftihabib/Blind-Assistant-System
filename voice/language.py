def build_alert(cls_name, distance_m, danger_level, position):
    dist_text = f"{distance_m} metres" if distance_m else "unknown distance"
    pos_text  = position.lower() if position else ""
    if danger_level == "DANGER":
        return f"Danger! {cls_name} very close, {dist_text}, on your {pos_text}!"
    elif danger_level == "WARNING":
        return f"Warning! {cls_name} ahead, {dist_text}, {pos_text}."
    elif danger_level == "CAUTION":
        return f"Caution! {cls_name} detected, {dist_text}, {pos_text}."
    else:
        return f"{cls_name} detected at {dist_text}."
