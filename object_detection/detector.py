import cv2
import numpy as np
import time
from object_detection.model_loader import load_model
from config import CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, KNOWN_WIDTHS, HIGH_PRIORITY, IMGSZ

class Detector:
    def __init__(self):
        self.model = load_model()
        self._frame_count = 0
        self._log_interval = 10   # print debug info every N detections

    def detect(self, frame):
        """Return list of detection dicts for ALL objects in frame"""
        detections = []
        self._frame_count += 1

        try:
            # Use augmented prediction for better detection accuracy
            results = self.model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                imgsz=IMGSZ,
                verbose=False,
                stream=False,
            )
            if not results:
                return detections

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                if self._frame_count % self._log_interval == 0:
                    print(f"[Detector] Frame {self._frame_count}: NO objects detected")
                return detections

            # ── bulk numpy extract ──────────────────
            boxes_xyxy  = result.boxes.xyxy.cpu().numpy()   # [N,4]
            confs       = result.boxes.conf.cpu().numpy()    # [N]
            cls_ids     = result.boxes.cls.cpu().numpy().astype(int)  # [N]
            names       = result.names                       # dict

            # ── DEBUG: log all raw detections ──────────
            if self._frame_count % self._log_interval == 0:
                raw_objects = []
                for i in range(len(boxes_xyxy)):
                    cls_name = names.get(cls_ids[i], "?")
                    raw_objects.append(f"{cls_name}({confs[i]:.0%})")
                print(f"[Detector] Frame {self._frame_count}: Found {len(boxes_xyxy)} objects → {', '.join(raw_objects)}")

            for i in range(len(boxes_xyxy)):
                x1,y1,x2,y2 = boxes_xyxy[i]
                conf         = float(confs[i])
                cls_id       = cls_ids[i]
                class_name   = names.get(cls_id, "object")

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w < 2 or bbox_h < 2:   # too small – skip noise
                    continue

                detections.append({
                    "class":       class_name,
                    "confidence":  conf,
                    "bbox":        (int(x1), int(y1), int(x2), int(y2)),
                    "bbox_width":  float(bbox_w),
                    "bbox_height": float(bbox_h),
                    "known_width": KNOWN_WIDTHS.get(class_name,
                                   KNOWN_WIDTHS["default"]),
                    "priority":    class_name in HIGH_PRIORITY,
                })

        except Exception as e:
            print(f"[Detector] Error: {e}")
            import traceback
            traceback.print_exc()

        return detections
