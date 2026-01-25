from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

class LiveChartWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.plot_widget = pg.PlotWidget()

        self.plot_widget.setBackground('#121212')
        self.plot_widget.setTitle("Population Analysis", color='#FFFFFF', size='12pt')
        self.plot_widget.setLabel('left', 'Cell Count', color='#FFFFFF')
        self.plot_widget.setLabel('bottom', 'Time (Frame)', color='#FFFFFF')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2))

        self.layout.addWidget(self.plot_widget)
        
    def update_chart(self, data_list):
        self.curve.setData(data_list)