import time

class FPSCounter:
    def __init__(self, avg=30):
        self.avg   = avg
        self.times = []

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.avg:
            self.times.pop(0)

    def get(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times)-1) / (self.times[-1] - self.times[0])
