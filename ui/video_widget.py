"""
Video Display Widget
Shows the microscope view with cell detection overlays
"""

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np


class VideoWidget(QWidget):
    """Widget for displaying video feed with detection overlays"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Video frame holder
        self.current_frame = None
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label
        self.title_label = QLabel("[ VİDEO EKRANI ]")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
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
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #00FF00;
            }
        """)
        self.video_label.setText("(Hücrelerin hareket ettiği\ngerçek mikroskop görüntüsü)\n\n(Biyolog Yeri)")
        self.video_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                font-family: 'Courier New', monospace;
                background-color: #000000;
                border: 2px solid #00FF00;
            }
        """)
        
        # Info label
        self.info_label = QLabel("(Biyolog Yeri)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 12px;
                font-family: 'Courier New', monospace;
                background-color: #1a1a1a;
                padding: 5px;
            }
        """)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
    
    def update_frame(self, frame):
        """
        Update the video display with a new frame
        
        Args:
            frame: OpenCV BGR image (numpy array)
        """
        if frame is None:
            return
        
        self.current_frame = frame
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get dimensions
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Convert to QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def clear_frame(self):
        """Clear the video display"""
        self.video_label.clear()
        self.video_label.setText("(Hücrelerin hareket ettiği\ngerçek mikroskop görüntüsü)\n\n(Biyolog Yeri)")
