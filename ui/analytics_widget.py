"""
Analytics Widget
Displays the live cell count chart
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from collections import deque


class AnalyticsWidget(QWidget):
    """Widget for displaying analytics chart"""
    
    def __init__(self, parent=None, max_points=200):
        super().__init__(parent)
        self.max_points = max_points
        self.data_buffer = deque(maxlen=max_points)
        
        # Initialize with zeros
        for _ in range(max_points):
            self.data_buffer.append(0)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("[ ANALYTICS ]")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                background-color: #1a1a1a;
                padding: 10px;
                border: 2px solid #00FF00;
            }
        """)
        layout.addWidget(title)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#000000')
        self.plot_widget.setLabel('left', 'Cell Count', color='#00FF00', size='12pt')
        self.plot_widget.setLabel('bottom', 'Time', color='#00FF00', size='12pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Style the axes
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#00FF00', width=2))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#00FF00', width=2))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#00FF00'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#00FF00'))
        
        # Set Y-axis range
        self.plot_widget.setYRange(0, 120, padding=0)
        
        # Create plot curve
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#00FFFF', width=3),
            symbol='o',
            symbolSize=8,
            symbolBrush='#00FFFF'
        )
        
        # Add border to plot widget
        self.plot_widget.setStyleSheet("""
            QWidget {
                border: 2px solid #00FF00;
                background-color: #000000;
            }
        """)
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def update_data(self, cell_count):
        """
        Update the chart with new cell count data
        
        Args:
            cell_count (int): Current cell count
        """
        self.data_buffer.append(cell_count)
        self.curve.setData(list(self.data_buffer))
    
    def clear_data(self):
        """Clear all data from the chart"""
        self.data_buffer.clear()
        for _ in range(self.max_points):
            self.data_buffer.append(0)
        self.curve.setData(list(self.data_buffer))
