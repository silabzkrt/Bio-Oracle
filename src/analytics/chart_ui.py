from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class LiveChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#121212')
        self.plot_widget.setTitle("Live Population Analysis", color='#FFFFFF', size='12pt')
        self.plot_widget.setLabel('left', 'Cell Count', color='#FFFFFF')
        self.plot_widget.setLabel('bottom', 'Time (Frames)', color='#FFFFFF')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        self.curve_real = self.plot_widget.plot(name='Real Data', pen=pg.mkPen('c', width=3))
        self.curve_pred = self.plot_widget.plot(name='AI Prediction', pen=pg.mkPen(color='#FFD700', width=3, style=Qt.PenStyle.DashLine))
        
        self.layout.addWidget(self.plot_widget)

    def update_chart(self, real_data, pred_data=None, pred_x_start=0):
        self.curve_real.setData(real_data)

        if pred_data is not None:
            x_values = np.arange(pred_x_start, pred_x_start + len(pred_data))
            self.curve_pred.setData(x=x_values, y=pred_data)
        else:
            self.curve_pred.clear()