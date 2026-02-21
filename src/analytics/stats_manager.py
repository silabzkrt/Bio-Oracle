import numpy as np
from collections import deque

class StatisticsManager:
    def __init__(self, max_history=200):
        self.history_buffer = deque(maxlen=max_history)

    def add_data_point(self, cell_count, toxicity=0.0):
        self.history_buffer.append({
            'cells': cell_count,
            'toxicity': toxicity
        })

    def get_cell_history(self):
        return [data['cells'] for data in self.history_buffer]

    def get_smoothed_history(self, window=10):
        raw_history = self.get_cell_history()
        if len(raw_history) < window:
            return raw_history
            
        weights = np.ones(window) / window
        smoothed = np.convolve(raw_history, weights, mode='valid')
        
        return list(raw_history[:window-1]) + list(smoothed)

    def get_ml_data(self):
        if len(self.history_buffer) < 2:
            return None, None
            
        y = np.array([data['cells'] for data in self.history_buffer])
        X_time = np.arange(len(y)).reshape(-1, 1)
        X_tox = np.array([data['toxicity'] for data in self.history_buffer]).reshape(-1, 1)
        
        X = np.hstack((X_time, X_tox))
        return X, y