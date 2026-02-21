import sys
import random
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from src.analytics.chart_ui import LiveChartWidget
from src.analytics.stats_manager import StatisticsManager
from src.analytics.predictor import PopulationPredictor

class AnalystTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Oracle: Analyst Module Test (Week 3)")
        self.resize(900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.chart = LiveChartWidget()
        layout.addWidget(self.chart)

        self.stats_manager = StatisticsManager(max_history=200)
        self.predictor = PopulationPredictor(future_steps=50)
        
        self.current_toxicity = 0.0 

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.run_simulation_step)
        self.timer.start()
        
        self.time_step = 0

    def run_simulation_step(self):
        if len(self.stats_manager.history_buffer) == 0:
            last_val = 50
        else:
            last_val = self.stats_manager.history_buffer[-1]['cells']
            
        if self.current_toxicity > 50:
            change = random.randint(-4, 0)
        else:
            change = random.randint(-2, 3)
            
        new_val = max(0, min(100, last_val + change))
        
        self.stats_manager.add_data_point(new_val, self.current_toxicity)
        self.time_step += 1

        real_history = self.stats_manager.get_smoothed_history(window=10)
        
        X, y = self.stats_manager.get_ml_data()
        future_predictions = self.predictor.predict_future(X, y, self.current_toxicity)
        
        self.chart.update_chart(
            real_data=real_history, 
            pred_data=future_predictions, 
            pred_x_start=len(real_history) - 1
        )
        
        if random.random() < 0.02:
            self.current_toxicity = 80.0
            print("TEST: Zehir %80'e fırladı! Grafiğin çökmesi lazım...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalystTestWindow()
    window.show()
    sys.exit(app.exec())