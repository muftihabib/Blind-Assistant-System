import cv2

from object_detection.model_loader import load_model
from object_detection.preprocess import Preprocessor
from config import CONFIDENCE, IOU_THRESHOLD, MAX_DETECTIONS, IMGSZ


class Detector:
    """YOLOv8n wrapper. Boxes are in the coordinates of the input (capture) frame."""

    def __init__(self, model_path=None, preprocess=True, conf=CONFIDENCE,
                 iou=IOU_THRESHOLD, imgsz=IMGSZ):
        self.model = load_model(model_path)
        self.pre = Preprocessor() if preprocess else None
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def predict_raw(self, frame, conf=None, max_det=MAX_DETECTIONS):
        """Return (Ultralytics result, scale): box coords * scale = capture-frame pixels."""
        h, w = frame.shape[:2]
        scale = 1.0
        if w > self.imgsz:
            scale = w / self.imgsz
            frame = cv2.resize(frame, (self.imgsz, round(h / scale)), interpolation=cv2.INTER_AREA)
        img = self.pre(frame) if self.pre else frame
        res = self.model.predict(source=img, conf=self.conf if conf is None else conf,
                                 iou=self.iou, max_det=max_det, imgsz=self.imgsz,
                                 verbose=False)[0]
        return res, scale

    def detect(self, frame):
        result, scale = self.predict_raw(frame)
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy() * scale
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 2 or bh < 2:
                continue
            detections.append({
                "class": names.get(int(cid), "object"),
                "class_id": int(cid),
                "confidence": float(conf),
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "bbox_width": bw,
                "bbox_height": bh,
            })
        return detections
