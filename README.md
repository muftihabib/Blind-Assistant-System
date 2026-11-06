```
vision_assistant/
│
├── main.py
├── requirements.txt
├── config.py
│
├── object_detection/
│   ├── __init__.py
│   ├── detector.py
│   ├── inference.py
│   ├── model_loader.py
│   ├── classes.py
│   └── utils.py
│
├── distance_estimation/
│   ├── __init__.py
│   ├── estimator.py
│   ├── monocular.py
│   ├── calibration.py
│   └── geometry.py
│
├── voice/
│   ├── __init__.py
│   ├── speaker.py
│   ├── alerts.py
│   ├── queue_manager.py
│   └── language.py
│
├── navigation/
│   ├── direction.py
│   ├── obstacle.py
│   └── guidance.py
│
├── camera/
│   ├── webcam.py
│   ├── video.py
│   └── stream.py
│
├── tracking/
│   ├── tracker.py
│   └── history.py
│
├── utils/
│   ├── draw.py
│   ├── fps.py
│   ├── logger.py
│   └── helpers.py
│
├── models/
│   └── rf_detr.pt
│
└── assets/
    └── sounds/

```