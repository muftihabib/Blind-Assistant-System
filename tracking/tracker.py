class Tracker:
    def __init__(self, max_missing=2):
        self.objects = {}
        self.next_id = 0
        self.max_missing = max_missing
        self.missing = {}

    def reset(self):
        self.objects.clear()
        self.missing.clear()
        self.next_id = 0

    def update(self, detections):
        if not detections:
            self.reset()
            return []

        tracked = []
        for oid, info in list(self.objects.items()):
            tracked.append((oid, info["centroid"], info.get("class_name"), info.get("bbox")))

        new_objects = {}
        used_indices = set()

        for oid, centroid, cls_name, bbox in tracked:
            best_idx = -1
            best_dist = float("inf")

            for i, det in enumerate(detections):
                if i in used_indices:
                    continue

                det_cls = det.get("class", "")
                det_center = ((det["bbox"][0] + det["bbox"][2]) // 2,
                              (det["bbox"][1] + det["bbox"][3]) // 2)
                dist = ((centroid[0] - det_center[0]) ** 2 + (centroid[1] - det_center[1]) ** 2) ** 0.5

                same_class = cls_name is None or det_cls == cls_name
                if same_class and dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0 and best_dist < 120:
                used_indices.add(best_idx)
                new_objects[oid] = {
                    "centroid": ((detections[best_idx]["bbox"][0] + detections[best_idx]["bbox"][2]) // 2,
                                 (detections[best_idx]["bbox"][1] + detections[best_idx]["bbox"][3]) // 2),
                    "class_name": detections[best_idx].get("class"),
                    "bbox": detections[best_idx]["bbox"],
                }
                self.missing[oid] = 0
            else:
                self.missing[oid] = self.missing.get(oid, 0) + 1
                if self.missing.get(oid, 0) > self.max_missing:
                    continue
                new_objects[oid] = {
                    "centroid": centroid,
                    "class_name": cls_name,
                    "bbox": bbox,
                }

        for oid in list(self.missing.keys()):
            if self.missing.get(oid, 0) > self.max_missing and oid not in new_objects:
                del self.missing[oid]

        for i, det in enumerate(detections):
            if i in used_indices:
                continue
            new_id = self.next_id
            self.next_id += 1
            new_objects[new_id] = {
                "centroid": ((det["bbox"][0] + det["bbox"][2]) // 2,
                             (det["bbox"][1] + det["bbox"][3]) // 2),
                "class_name": det.get("class"),
                "bbox": det["bbox"],
            }
            self.missing[new_id] = 0

        self.objects = new_objects
        keys = list(self.objects.keys())
        for i, det in enumerate(detections):
            det["track_id"] = -1
            for oid, info in self.objects.items():
                if info["bbox"] == det["bbox"]:
                    det["track_id"] = oid
                    break
            if det["track_id"] == -1 and i < len(keys):
                det["track_id"] = keys[i]

        return detections
