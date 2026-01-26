import sys
import random
from collections import deque
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from src.analytics.chart_ui import LiveChartWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bio-Oracle: Analyst Test")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.chart_monitor = LiveChartWidget()
        layout.addWidget(self.chart_monitor)

        self.data_buffer = deque(maxlen=200)
        for _ in range(200):
            self.data_buffer.append(0)

        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start()

    def update_simulation(self):
        fake_cell_count = random.randint(20, 80)
        self.data_buffer.append(fake_cell_count)
        self.chart_monitor.update_chart(list(self.data_buffer))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())