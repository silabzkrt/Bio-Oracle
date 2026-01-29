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

from .video_widget import VideoWidget
from .control_panel import ControlPanel
from .analytics_widget import AnalyticsWidget

# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules import CellDetector, CellTracker, DataLogger
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
        
        # Open new video
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {video_path}")
            return
        
        self.current_video_path = video_path
        self.frame_count = 0
        
        # Initialize detection modules if available
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
        
        # Clear analytics
        self.analytics_widget.clear_data()
        
        # Start playback
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
        """Update video frame and perform detection"""
        if not self.video_capture or not self.video_capture.isOpened():
            self.stop_video()
            return
        
        ret, frame = self.video_capture.read()
        
        if not ret:
            # Video ended, loop back
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            return
        
        self.frame_count += 1
        
        # Perform detection if modules available
        cell_count = 0
        
        if self.detector and self.tracker:
            try:
                # Detect cells
                detections = self.detector.detect(frame)
                
                # Track cells
                tracked_detections = self.tracker.update(detections)
                
                # Apply environmental effects (simulation)
                tracked_detections = self.apply_environmental_effects(tracked_detections)
                
                # Draw detections on frame
                for det in tracked_detections:
                    x1, y1, x2, y2 = det['bbox']
                    status = det.get('status', 'unknown')
                    
                    # Choose color based on status
                    if status == 'moving':
                        color = config.COLOR_MOVING
                    elif status == 'staying':
                        color = config.COLOR_STAYING
                    else:
                        color = config.COLOR_UNKNOWN
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
                    
                    # Draw label
                    label = f"ID:{det['track_id']} {status}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        config.FONT, config.FONT_SCALE, color, config.FONT_THICKNESS
                    )
                
                cell_count = len(tracked_detections)
                
                # Log data
                if config.ENABLE_LOGGING and self.logger:
                    counts = self.tracker.get_counts()
                    self.logger.log_counts(self.frame_count, counts, len(detections))
                
            except Exception as e:
                print(f"Detection error: {e}")
                cell_count = 0
        else:
            # Demo mode - simulate cell count
            import random
            cell_count = random.randint(20, 80)
        
        # Update display
        self.video_widget.update_frame(frame)
        
        # Update analytics
        self.analytics_widget.update_data(cell_count)
        
        # Update status bar
        self.status_bar.showMessage(
            f"Frame: {self.frame_count} | Cells: {cell_count} | "
            f"Toxicity: {self.toxicity}% | Temp: {self.temperature}°C"
        )
    
    def apply_environmental_effects(self, detections):
        """
        Simulate environmental effects on cells
        This is a placeholder for future simulation logic
        """
        # TODO: Implement actual simulation based on toxicity and temperature
        # For now, just return detections as-is
        return detections
    
    def on_toxicity_changed(self, value):
        """Handle toxicity slider change"""
        self.toxicity = value
    
    def on_temperature_changed(self, value):
        """Handle temperature slider change"""
        self.temperature = value
    
    def on_kill_button_clicked(self):
        """Handle kill button click"""
        reply = QMessageBox.question(
            self,
            'Confirm',
            'Are you sure you want to kill all cells?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # TODO: Implement kill logic
            self.status_bar.showMessage("⚠️ Kill function activated!")
            QMessageBox.information(self, "Info", "Kill function activated! (Simulation)")
    
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
