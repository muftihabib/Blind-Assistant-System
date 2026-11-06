from voice.language      import build_alert
from voice.queue_manager import QueueManager

class AlertSystem:
    def __init__(self, speaker, cooldown=3):
        self.speaker = speaker
        self.queue   = QueueManager(cooldown)

    def process(self, detections):
        priority_order = {"DANGER": 0, "WARNING": 1,
                          "CAUTION": 2, "SAFE": 3, "UNKNOWN": 4}
        urgent = sorted(
            [d for d in detections if d.get("distance_m") is not None],
            key=lambda d: (
                priority_order.get(d["danger_level"], 9),
                d["distance_m"]
            )
        )
        for det in urgent:
            cls   = det["class"]
            level = det["danger_level"]
            if level in ("DANGER", "WARNING", "CAUTION"):
                if self.queue.can_speak(cls):
                    msg = build_alert(
                        cls,
                        det["distance_m"],
                        level,
                        det.get("position", "")
                    )
                    self.speaker.say(msg)
                    print(f"Alert: {msg}")
                    break
