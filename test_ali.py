import sys
import random
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from src.analytics.chart_ui import LiveChartWidget
from src.analytics.stats_manager import StatisticsManager

class AnalystTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bio-Oracle: Analyst Module Test (Week 2)")
        self.resize(900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.chart = LiveChartWidget()
        layout.addWidget(self.chart)

        self.stats_manager = StatisticsManager(max_history=200)

        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.run_simulation_step)
        self.timer.start()

    def run_simulation_step(self):
        last_val = self.stats_manager.get_summary_stats()['current']
        change = random.randint(-2, 2)
        new_val = max(0, min(100, last_val + change))
        
        self.stats_manager.add_data_point(new_val)

        history_data = self.stats_manager.get_history()
        
        if random.random() < 0.05:
            X, y = self.stats_manager.get_ml_data()
            print(f"DEBUG: ML Data Shape -> X: {X.shape}, y: {y.shape}")

        self.chart.update_chart(real_data=history_data, pred_data=None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalystTestWindow()
    window.show()
    sys.exit(app.exec())