import numpy as np
from sklearn.linear_model import LinearRegression

class PopulationPredictor:
    def __init__(self, future_steps=50, lookback_window=30):
        self.model = LinearRegression()
        self.future_steps = future_steps
        self.lookback = lookback_window 

    def predict_future(self, X, y, current_toxicity):
        if X is None or len(y) < 10:
            return None
        
        X_recent = X[-self.lookback:]
        y_recent = y[-self.lookback:]
        
        self.model.fit(X_recent, y_recent)
        
        last_time_step = X[-1][0]
        future_time = np.arange(last_time_step + 1, last_time_step + 1 + self.future_steps).reshape(-1, 1)
        future_tox = np.full((self.future_steps, 1), current_toxicity)
        future_X = np.hstack((future_time, future_tox))
        
        future_predictions = self.model.predict(future_X)
        future_predictions = np.maximum(future_predictions, 0)
        
        return future_predictions