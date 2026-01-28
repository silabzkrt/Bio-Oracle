"""
Control Panel Widget
Provides controls for toxicity, temperature, and kill button
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """Control panel for environmental parameters"""
    
    # Signals
    toxicity_changed = pyqtSignal(int)
    temperature_changed = pyqtSignal(int)
    kill_button_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("[ KONTROL PANELİ ]")
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
        
        # Control group
        control_group = QGroupBox()
        control_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a1a;
                border: 2px solid #00FF00;
                margin-top: 10px;
            }
        """)
        
        control_layout = QVBoxLayout()
        control_layout.setSpacing(20)
        
        # Toxicity slider (Zehir Miktarı)
        toxicity_label = QLabel("• Zehir Miktarı")
        toxicity_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        self.toxicity_slider = QSlider(Qt.Orientation.Horizontal)
        self.toxicity_slider.setMinimum(0)
        self.toxicity_slider.setMaximum(100)
        self.toxicity_slider.setValue(0)
        self.toxicity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #00FF00;
                height: 8px;
                background: #000000;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #00FF00;
                border: 1px solid #00FF00;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.toxicity_slider.valueChanged.connect(self.on_toxicity_changed)
        
        self.toxicity_value_label = QLabel("0%")
        self.toxicity_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toxicity_value_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 12px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        control_layout.addWidget(toxicity_label)
        control_layout.addWidget(self.toxicity_slider)
        control_layout.addWidget(self.toxicity_value_label)
        
        # Temperature slider (Sıcaklık Slider'ı)
        temp_label = QLabel("• Sıcaklık Slider'ı")
        temp_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setMinimum(-20)
        self.temperature_slider.setMaximum(50)
        self.temperature_slider.setValue(25)
        self.temperature_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #00FF00;
                height: 8px;
                background: #000000;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #00FF00;
                border: 1px solid #00FF00;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.temperature_slider.valueChanged.connect(self.on_temperature_changed)
        
        self.temperature_value_label = QLabel("25°C")
        self.temperature_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.temperature_value_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 12px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        control_layout.addWidget(temp_label)
        control_layout.addWidget(self.temperature_slider)
        control_layout.addWidget(self.temperature_value_label)
        
        # Kill button (Öldür Butonu)
        kill_label = QLabel("• Öldür Butonu")
        kill_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
        """)
        
        self.kill_button = QPushButton("ÖLDÜR")
        self.kill_button.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                padding: 10px;
                border: 2px solid #FF0000;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #CC0000;
            }
            QPushButton:pressed {
                background-color: #990000;
            }
        """)
        self.kill_button.clicked.connect(self.on_kill_clicked)
        
        control_layout.addWidget(kill_label)
        control_layout.addWidget(self.kill_button)
        
        # Info label
        info_label = QLabel("(Kimyager Yeri)")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 12px;
                font-family: 'Courier New', monospace;
                padding: 10px;
            }
        """)
        control_layout.addWidget(info_label)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        self.setLayout(layout)
    
    def on_toxicity_changed(self, value):
        """Handle toxicity slider change"""
        self.toxicity_value_label.setText(f"{value}%")
        self.toxicity_changed.emit(value)
    
    def on_temperature_changed(self, value):
        """Handle temperature slider change"""
        self.temperature_value_label.setText(f"{value}°C")
        self.temperature_changed.emit(value)
    
    def on_kill_clicked(self):
        """Handle kill button click"""
        self.kill_button_clicked.emit()
