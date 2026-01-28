"""
Configuration file for Bio-Oracle
All settings, thresholds, paths, and colors are defined here
"""

import os

# ============================================================================
# PATH SETTINGS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "best.pt")

# Input/Output paths
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "assets", "input_videos")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================================
# DETECTION SETTINGS
# ============================================================================
# YOLO confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Image size for inference
IMG_SIZE = 640

# Device to use ('cpu' or 'cuda' or '0' for GPU)
DEVICE = 'cpu'

# ============================================================================
# TRACKING SETTINGS
# ============================================================================
# Distance threshold for determining if a cell is "moving" (in pixels)
MOVEMENT_THRESHOLD = 50

# Number of frames to track before classifying as "staying"
STAYING_FRAME_COUNT = 30

# Maximum frames to keep in tracking history
MAX_TRACKING_HISTORY = 100

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Colors (BGR format for OpenCV)
COLOR_MOVING = (0, 255, 0)      # Green
COLOR_STAYING = (0, 0, 255)     # Red
COLOR_UNKNOWN = (255, 0, 0)     # Blue

# Bounding box thickness
BOX_THICKNESS = 2

# Text settings
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
# Enable/disable logging
ENABLE_LOGGING = True

# Log file format
LOG_DATE_FORMAT = "%Y-%m-%d"
LOG_TIME_FORMAT = "%H:%M:%S"
