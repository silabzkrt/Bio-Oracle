import numpy as np
from collections import deque

class StatisticsManager:
    def __init__(self, max_history=200):
        self.history_buffer = deque(maxlen=max_history)
        
        for _ in range(max_history):
            self.history_buffer.append(0)

    def add_data_point(self, value):
        self.history_buffer.append(value)

    def get_history(self):
        return list(self.history_buffer)

    def get_summary_stats(self):
        data = np.array(self.history_buffer)
        return {
            "current": data[-1],
            "mean": np.mean(data),
            "max": np.max(data)
        }

    def get_ml_data(self):
        y = np.array(self.history_buffer)
        X = np.arange(len(y))
        X = X.reshape(-1, 1)
        
        return X, y