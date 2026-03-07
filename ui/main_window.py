"""
Main Window for Bio-Oracle Application
Integrates video display, control panel, and analytics
"""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QMessageBox, QStatusBar)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction
import cv2
import os
import random

from .video_widget import VideoWidget
from .control_panel import ControlPanel
from .analytics_widget import AnalyticsWidget


# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.detector import CellDetector
    from modules.tracker import CellTracker
    from modules.logger import DataLogger
    import config
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Warning: Detection modules not available. Running in demo mode.")

class BioOracleWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Oracle v1.0")
        self.setMinimumSize(1200, 800)
        
        # State variables
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.current_video_path = None
        
        # Detection components (if available)
        self.detector = None
        self.tracker = None
        self.logger = None
        self.frame_count = 0
        
        # Environmental parameters
        self.toxicity = 0
        self.temperature = 25
        
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_connections()
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QMenuBar {
                background-color: #2a2a2a;
                color: #00FF00;
                font-family: 'Courier New', monospace;
            }
            QMenuBar::item:selected {
                background-color: #3a3a3a;
            }
            QMenu {
                background-color: #2a2a2a;
                color: #00FF00;
                font-family: 'Courier New', monospace;
            }
            QMenu::item:selected {
                background-color: #3a3a3a;
            }
            QStatusBar {
                background-color: #2a2a2a;
                color: #00FF00;
                font-family: 'Courier New', monospace;
            }
        """)
    
    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Top section: Video and Control Panel
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        # Video widget (left side)
        self.video_widget = VideoWidget()
        top_layout.addWidget(self.video_widget, 2)
        
        # Control panel (right side)
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(400)
        top_layout.addWidget(self.control_panel, 1)
        
        main_layout.addLayout(top_layout, 2)
        
        # Bottom section: Analytics
        self.analytics_widget = AnalyticsWidget(max_points=200)
        self.analytics_widget.setMaximumHeight(300)
        main_layout.addWidget(self.analytics_widget, 1)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open Video', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Control menu
        control_menu = menubar.addMenu('&Control')
        
        play_action = QAction('&Play/Pause', self)
        play_action.setShortcut('Space')
        play_action.triggered.connect(self.toggle_playback)
        control_menu.addAction(play_action)
        
        stop_action = QAction('&Stop', self)
        stop_action.triggered.connect(self.stop_video)
        control_menu.addAction(stop_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.control_panel.toxicity_changed.connect(self.on_toxicity_changed)
        self.control_panel.temperature_changed.connect(self.on_temperature_changed)
        self.control_panel.kill_button_clicked.connect(self.on_kill_button_clicked)
    
    def open_video(self):
        """Open a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        """Load a video file"""
        # Stop current video if playing
        if self.is_playing:
            self.stop_video()
        
        # Release previous capture
        if self.video_capture:
            self.video_capture.release()
        
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {video_path}")
            return
        
        self.current_video_path = video_path
        self.frame_count = 0
        
        if MODULES_AVAILABLE:
            try:
                self.detector = CellDetector(
                    model_path=config.MODEL_PATH,
                    confidence_threshold=config.CONFIDENCE_THRESHOLD,
                    device=config.DEVICE
                )
                self.tracker = CellTracker(
                    movement_threshold=config.MOVEMENT_THRESHOLD,
                    staying_frame_count=config.STAYING_FRAME_COUNT,
                    max_history=config.MAX_TRACKING_HISTORY
                )
                self.logger = DataLogger(
                    logs_dir=config.LOGS_DIR,
                    date_format=config.LOG_DATE_FORMAT,
                    time_format=config.LOG_TIME_FORMAT
                )
                self.status_bar.showMessage(f"Loaded: {os.path.basename(video_path)} | Detection: Active")
            except Exception as e:
                print(f"Error initializing detection: {e}")
                self.detector = None
                self.tracker = None
                self.status_bar.showMessage(f"Loaded: {os.path.basename(video_path)} | Detection: Inactive")
        else:
            self.status_bar.showMessage(f"Loaded: {os.path.basename(video_path)} | Demo Mode")
        
        self.analytics_widget.clear_data()
        
        self.start_playback()
    
    def start_playback(self):
        """Start video playback"""
        if self.video_capture and self.video_capture.isOpened():
            self.is_playing = True
            self.timer.start(33)  # ~30 FPS
    
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def pause_playback(self):
        """Pause video playback"""
        self.is_playing = False
        self.timer.stop()
    
    def stop_video(self):
        """Stop video playback"""
        self.is_playing = False
        self.timer.stop()
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.video_widget.clear_frame()
        self.frame_count = 0
        self.status_bar.showMessage("Stopped")

    def update_frame(self):
            """Update video frame and perform detection with Environmental Stress"""
            if not self.video_capture or not self.video_capture.isOpened():
                self.stop_video()
                return
            
            ret, frame = self.video_capture.read()
            
            if not ret:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                return
            
            self.frame_count += 1
            cell_count = 0
            
            if self.detector and self.tracker:
                try:
                    detections = self.detector.detect(frame)
                    tracked_detections = self.tracker.update(detections)

                    surviving_detections = self.apply_environmental_effects(tracked_detections)
                    cell_count = len(surviving_detections)

                    import random
                    for det in surviving_detections:
                        x1, y1, x2, y2 = det['bbox']
                        
                        jitter_range = max(0, int((self.temperature - 25) / 5))
                        if jitter_range > 0:
                            x1 += random.randint(-jitter_range, jitter_range)
                            y1 += random.randint(-jitter_range, jitter_range)
                            x2 += random.randint(-jitter_range, jitter_range)
                            y2 += random.randint(-jitter_range, jitter_range)

                        status = det.get('status', 'unknown')
                        color = config.COLOR_MOVING if status == 'moving' else config.COLOR_STAYING
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
                        cv2.putText(frame, f"ID:{det['track_id']} {status}", (x1, y1 - 10),
                                    config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS)
                    
                    if config.ENABLE_LOGGING and self.logger:
                        counts = self.tracker.get_counts() 
                        self.logger.log_counts(self.frame_count, counts, len(detections), self.toxicity, self.temperature)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                    cell_count = 0
            else:
                import random
                cell_count = random.randint(20, 80)
            
            self.video_widget.update_frame(frame)
            self.analytics_widget.update_data(cell_count)
            
            status_text = f"Frame: {self.frame_count} | Cells: {cell_count} | Toxicity: {self.toxicity}% | Temp: {self.temperature}°C"
            self.status_bar.showMessage(status_text)

            if self.toxicity > 75:
                self.status_bar.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold;")
            elif self.toxicity > 40:
                self.status_bar.setStyleSheet("background-color: #ffff00; color: black;")
            else:
                self.status_bar.setStyleSheet("")

    def apply_environmental_effects(self, detections):
        """
        Chemist Logic: Simulates cell death based on toxicity levels.
        """
        tox_level = self.toxicity 
        
        survivors = []
        for det in detections:
            death_chance = (tox_level / 333.3) 
            
            if random.random() > death_chance:
                survivors.append(det)
                
        return survivors
    
    def on_toxicity_changed(self, value):
        """Handle toxicity slider change and reset UI warnings"""
        self.toxicity = value
        self.status_bar.setStyleSheet("")        
        self.status_bar.showMessage(f"Toxicity adjusted to {value}%")
    
    def on_temperature_changed(self, value):
        """Handle temperature slider change"""
        self.temperature = value

    def on_kill_button_clicked(self):
        """Handle kill button click - Chemist Logic"""
        reply = QMessageBox.question(
            self,
            'Confirm Intervention',
            'Are you sure you want to release the neutralizing agent and kill all cells?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.tracker:
                self.tracker.tracks = {} 
            
            self.status_bar.showMessage("Biological Wipe Complete: 0 Cells remaining.")
            self.status_bar.setStyleSheet("background-color: #770000; color: white; font-weight: bold;")
            
            if self.analytics_widget:
                self.analytics_widget.clear_data()
            
            QMessageBox.information(self, "Protocol X", "Neutralizing agent released. All tracks cleared.")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Bio-Oracle",
            "<h2>Bio-Oracle v1.0</h2>"
            "<p>Real-time cell detection and tracking system</p>"
            "<p><b>Modules:</b></p>"
            "<ul>"
            "<li>Detector: Cell detection and visualization</li>"
            "<li>Tracker: Movement tracking</li>"
            "<li>Logger: Data analysis and logging</li>"
            "</ul>"
            "<p>© 2026 Bio-Oracle Project</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_capture:
            self.video_capture.release()
        event.accept()
