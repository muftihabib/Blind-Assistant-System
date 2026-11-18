"""
Detector back-ends for tools/eval_benchmark.py.

Every back-end exposes
    names                 {class_index: name}  — COCO 80-class indices (YOLO order)
    predict(img, conf)    -> [(x1, y1, x2, y2), score, class_index] in image pixels

    yolo : any Ultralytics model (YOLOv8n, YOLOv5nu, YOLOv3-tinyu, ...) via Detector
    ssd  : TensorFlow SSD-MobileNetV2 (COCO) through OpenCV DNN, 300x300 input

The same pre-processing (denoise + CLAHE) is applied to every back-end unless
--no-preprocess is given.
"""

import cv2

from object_detection.detector import Detector
from object_detection.preprocess import Preprocessor


class YoloBackend:
    def __init__(self, model, preprocess, nms_iou, imgsz):
        self.det = Detector(model, preprocess=preprocess, iou=nms_iou, imgsz=imgsz)
        self.names = dict(self.det.model.names)
        self.input_desc = f"{imgsz}x{imgsz} letterbox"

    def predict(self, img, conf):
        res, scale = self.det.predict_raw(img, conf=conf, max_det=300)
        out = []
        if res.boxes is not None:
            for b, c, ci in zip(res.boxes.xyxy.cpu().numpy() * scale,
                                res.boxes.conf.cpu().numpy(),
                                res.boxes.cls.cpu().numpy().astype(int)):
                out.append((tuple(map(float, b)), float(c), int(ci)))
        return out


class SsdBackend:
    """
    TF Object Detection API ssd_mobilenet_v2_coco_2018_03_29 via cv2.dnn.
    The .pbtxt must have DetectionOutput set to nms_threshold 0.45,
    confidence_threshold 0.001, keep_top_k 300 to match the YOLO protocol.
    """

    def __init__(self, pb, pbtxt, preprocess, size=300):
        from ultralytics.data.converter import coco91_to_coco80_class
        from ultralytics.utils import YAML
        from ultralytics.utils import ROOT
        self.net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
        self.map91 = coco91_to_coco80_class()          # index: coco_id - 1 -> 80-class index
        self.names = YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
        self.pre = Preprocessor() if preprocess else None
        self.size = size
        self.input_desc = f"{size}x{size} resize (native SSD input)"

    def predict(self, img, conf):
        h, w = img.shape[:2]
        x = self.pre(img) if self.pre else img
        self.net.setInput(cv2.dnn.blobFromImage(x, size=(self.size, self.size), swapRB=True, crop=False))
        out = []
        for _, label, score, x1, y1, x2, y2 in self.net.forward()[0, 0]:
            if score < conf:
                continue
            lab = int(label)
            if lab < 1 or lab > len(self.map91) or self.map91[lab - 1] is None:
                continue
            out.append(((x1 * w, y1 * h, x2 * w, y2 * h), float(score), int(self.map91[lab - 1])))
        return out
