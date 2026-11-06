COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

KNOWN_WIDTHS = {
    "person": 45, "bicycle": 60, "car": 180, "motorcycle": 80,
    "bus": 260, "truck": 260, "traffic light": 30, "stop sign": 75,
    "bench": 150, "bird": 30, "cat": 40, "dog": 60, "horse": 150,
    "backpack": 35, "umbrella": 90, "handbag": 35, "suitcase": 60,
    "bottle": 7, "cup": 8, "chair": 50, "couch": 180, "bed": 140,
    "dining table": 120, "toilet": 40, "tv": 80, "laptop": 35,
    "cell phone": 8, "keyboard": 45, "book": 20, "clock": 25,
    "refrigerator": 70, "sink": 55, "microwave": 50,
    "default": 40,
}

HIGH_PRIORITY = {
    "person", "car", "truck", "bus", "motorcycle",
    "bicycle", "dog", "traffic light", "stop sign"
}

# DANGER_COLORS = {
#     "DANGER":  (0,   0,   255),
#     "WARNING": (0,  128,  255),
#     "CAUTION": (0,  255,  255),
#     "SAFE":    (0,  255,    0),
#     "UNKNOWN": (128,128,  128),
# }
