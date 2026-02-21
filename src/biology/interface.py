"""
Biologist Widget - Video Display with Cell Detection
Embeds OpenCV processing into PyQt6 using QThread
"""

from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import numpy as np
from .detector import CellDetector


class VideoThread(QThread):
    """Thread for processing video frames without blocking the UI"""
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    cell_count_signal = pyqtSignal(int)
    video_finished_signal = pyqtSignal()
    
    def __init__(self, use_yolo=False, model_path=None):
        super().__init__()
        self.video_path = None
        self.is_running = True
        self.is_paused = False
        self.detector = CellDetector(use_yolo=use_yolo, model_path=model_path)
        self.use_detection = True
    
    def set_video_path(self, path):
        """Set the video file path"""
        self.video_path = path
    
    def toggle_detection(self, enabled):
        """Enable/disable cell detection"""
        self.use_detection = enabled
    
    def pause(self):
        """Pause video playback"""
        self.is_paused = True
    
    def resume(self):
        """Resume video playback"""
        self.is_paused = False
    
    def stop(self):
        """Stop the thread"""
        self.is_running = False
        self.wait()
    
    def run(self):
        """Main thread loop - processes video frames"""
        if not self.video_path:
            return
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        while self.is_running:
            if self.is_paused:
                self.msleep(100)
                continue
            
            ret, frame = cap.read()
            
            if ret:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (1024, 768))
                
                # Apply cell detection if enabled
                if self.use_detection:
                    processed_frame, locked_count, candidate_count = self.detector.process(frame)
                    cell_count = self.detector.get_cell_count()
                    self.cell_count_signal.emit(cell_count)
                else:
                    processed_frame = frame
                    self.cell_count_signal.emit(0)
                
                # Convert BGR to RGB for PyQt
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                self.change_pixmap_signal.emit(rgb_image)
                
                # Control playback speed (30 FPS ≈ 33ms)
                self.msleep(33)
            else:
                # Video finished - loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Or emit finished signal
                # self.video_finished_signal.emit()
                # break
        
        cap.release()


class BiologistWidget(QWidget):
    """
    Main widget for the Biologist module
    Displays video with cell detection in PyQt6 interface
    """
    
    def __init__(self, use_yolo=False, model_path=None):
        super().__init__()
        self.setWindowTitle("Bio-Oracle: The Biologist" + (" [YOLO11]" if use_yolo else ""))
        self.display_width = 800
        self.display_height = 600
        self.use_yolo = use_yolo
        self.model_path = model_path
        
        # Video thread
        self.thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Title label
        title = QLabel("[ VIDEO SCREEN - Cell Detection ]")
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
        
        # Video display label
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #00FF00;
            }
        """)
        self.image_label.setText("Click 'Load Video' to start\n\n(Real microscope view with cell detection)")
        layout.addWidget(self.image_label)
        
        # Info label (cell count)
        self.info_label = QLabel("Cells Detected: 0")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                font-family: 'Courier New', monospace;
                background-color: #1a1a1a;
                padding: 5px;
            }
        """)
        layout.addWidget(self.info_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #00FF00;
                color: #000000;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #00FF00;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
        """)
        button_layout.addWidget(self.load_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: #000000;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #FFA500;
            }
            QPushButton:hover {
                background-color: #FF8C00;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        button_layout.addWidget(self.pause_btn)
        
        self.detection_btn = QPushButton("Detection: ON")
        self.detection_btn.clicked.connect(self.toggle_detection)
        self.detection_btn.setEnabled(False)
        self.detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #00FF00;
                color: #000000;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #00FF00;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        button_layout.addWidget(self.detection_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
            }
        """)
    
    def load_video(self):
        """Load a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            # Stop existing thread if running
            if self.thread:
                self.thread.stop()
            
            # Create and start new thread with YOLO support
            self.thread = VideoThread(use_yolo=self.use_yolo, model_path=self.model_path)
            self.thread.set_video_path(file_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.cell_count_signal.connect(self.update_cell_count)
            self.thread.start()
            
            # Enable buttons
            self.pause_btn.setEnabled(True)
            self.detection_btn.setEnabled(True)
            
            self.info_label.setText("Video loaded - Processing...")
    
    def toggle_pause(self):
        """Pause/resume video playback"""
        if self.thread:
            if self.pause_btn.text() == "Pause":
                self.thread.pause()
                self.pause_btn.setText("Resume")
            else:
                self.thread.resume()
                self.pause_btn.setText("Pause")
    
    def toggle_detection(self):
        """Toggle cell detection on/off"""
        if self.thread:
            if self.detection_btn.text() == "Detection: ON":
                self.thread.toggle_detection(False)
                self.detection_btn.setText("Detection: OFF")
                self.detection_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FF0000;
                        color: #FFFFFF;
                        font-weight: bold;
                        padding: 10px;
                        border: 2px solid #FF0000;
                    }
                    QPushButton:hover {
                        background-color: #CC0000;
                    }
                """)
            else:
                self.thread.toggle_detection(True)
                self.detection_btn.setText("Detection: ON")
                self.detection_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #00FF00;
                        color: #000000;
                        font-weight: bold;
                        padding: 10px;
                        border: 2px solid #00FF00;
                    }
                    QPushButton:hover {
                        background-color: #00CC00;
                    }
                """)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Update the display with new frame"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    @pyqtSlot(int)
    def update_cell_count(self, count):
        """Update cell count display"""
        self.info_label.setText(f"Cells Detected: {count}")
    
    def convert_cv_qt(self, cv_img):
        """Convert OpenCV image to QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.display_width, 
            self.display_height, 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        return QPixmap.fromImage(p)
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.thread:
            self.thread.stop()
        event.accept()
