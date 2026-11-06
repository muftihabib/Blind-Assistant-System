import time

class QueueManager:
    def __init__(self, cooldown_seconds=3):
        self.cooldown   = cooldown_seconds
        self._last_time = {}

    def can_speak(self, class_name):
        now  = time.time()
        last = self._last_time.get(class_name, 0)
        if now - last >= self.cooldown:
            self._last_time[class_name] = now
            return True
        return False

    def reset(self, class_name=None):
        if class_name:
            self._last_time.pop(class_name, None)
        else:
            self._last_time.clear()
