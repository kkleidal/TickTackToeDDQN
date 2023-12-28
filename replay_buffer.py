import numpy as np


class CircularBuffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = []
        self.idx = 0

    def append(self, x):
        if len(self.buffer) < self.max_len:
            self.buffer.append(x)
        else:
            self.buffer[self.idx] = x
            self.idx = (self.idx + 1) % self.max_len

    def sample(self, n):
        return [
            self.buffer[i]
            for i in np.random.choice(len(self.buffer), size=n, replace=False).tolist()
        ]

    def __len__(self):
        return len(self.buffer)
