# 🧬 Bio-Oracle: Cell Detection & Tracking System

![Python](https://img.shields.io/badge/Language-Python_3.10+-blue.svg?style=for-the-badge&logo=python)
![YOLO11](https://img.shields.io/badge/AI-YOLO11-yellow.svg?style=for-the-badge&logo=ultralytics)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg?style=for-the-badge&logo=opencv)

<p align="center">
  <em>A Python-based cell detection and tracking system using YOLO11 and OpenCV.</em>
</p>

---

## 📖 Overview

**Bio-Oracle** is a real-time cell detection and tracking platform that analyzes microscopic video feeds to detect, track, and classify cell movements.

Built with **Python**, the system uses **YOLO11** for cell detection, custom tracking algorithms to classify cells as "Moving" or "Staying", and comprehensive logging to record all detection and movement data.

---

## 🚀 Key Features

* **🦠 Cell Detection:** Real-time object detection using YOLO11 with configurable confidence thresholds
* **📍 Movement Tracking:** Intelligent tracking system that classifies cells as "Moving" vs "Staying" based on position history
* **📊 Data Logging:** Comprehensive logging system that records:
  - Frame-by-frame cell counts
  - Individual cell movement patterns
  - Processing summaries and statistics
* **🎨 Visual Feedback:** Color-coded bounding boxes (Green=Moving, Red=Staying, Blue=Unknown)
* **⚙️ Configurable:** All settings centralized in `config.py` for easy customization

---

## 📁 Project Structure

```
Bio-Oracle/
│
├── app.py                   # Application entry point - Run this to start
├── config.py                # Main configuration settings
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── LICENSE                  # License file
├── logo.png                 # Application logo
│
├── src/                     # Source code (all modules)
│   ├── analytics/           # Analytics and charting
│   │   ├── chart_ui.py
│   │   ├── predictor.py
│   │   └── stats_manager.py
│   ├── biology/             # Biological entity detection
│   │   ├── detector.py
│   │   └── interface.py
│   ├── chemistry/           # Chemistry simulation logic
│   │   ├── controls.py
│   │   └── logic.py
│   ├── core/                # Core detection & tracking modules
│   │   ├── bio_entity.py
│   │   ├── detector.py
│   │   ├── logger.py
│   │   ├── tracker.py
│   │   ├── traditional_detector.py
│   │   └── vision_manager.py
│   ├── simulation/          # Simulation math engine
│   │   └── math_engine.py
│   ├── ui/                  # User interface components
│   │   ├── main_window.py
│   │   ├── video_widget.py
│   │   ├── control_panel.py
│   │   └── analytics_widget.py
│   └── utils/               # Utility functions
│       ├── data_logger.py
│       └── visualizer.py
│
├── configs/                 # Additional configuration files
│   ├── config.yaml
│   └── high_sensitivity_config.py
│
├── scripts/                 # Analysis and utility scripts
│   ├── analyze_cells.py
│   ├── analyze_per_second.py
│   └── count_cells_per_frame.py
│
├── tests/                   # Test files and experiments
│   ├── test_ali.py
│   └── computer_vision/     # Computer vision tests
│
├── assets/                  # Static files
│   ├── input_videos/        # Place test videos here
│   └── models/              # Pre-trained models
│
├── data/                    # Data files and outputs
│   ├── input_samples/
│   └── outputs/             # Generated analysis results
│
├── models/                  # YOLO model files
│   └── yolo11n.pt
│
└── training_workspace/      # Model training environment
    ├── train_model.py
    ├── data.yaml
    └── dataset/             # Training images/labels
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for faster inference)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Bio-Oracle
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your model:**
   - Place your trained YOLO11 model at `assets/models/best.pt`
   - Or train a new model (see Training section below)

5. **Add test videos:**
   - Place video files in `assets/input_videos/`
   - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

---

## 🚀 Usage

### Run Detection on Video

```bash
# Process first video found in input_videos/
python main.py

# Or specify a video file
python main.py path/to/your/video.mp4
```

### Controls
- Press `q` to quit during processing
- Results are saved to `logs/` directory

### Output Files
- `logs/YYYY-MM-DD_counts.txt` - Cell count summaries per frame
- `logs/YYYY-MM-DD_movement.txt` - Detailed movement analysis per cell

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Detection settings
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detections
DEVICE = 'cpu'                # 'cpu', 'cuda', or '0' for GPU

# Tracking settings
MOVEMENT_THRESHOLD = 50       # Pixels to classify as "moving"
STAYING_FRAME_COUNT = 30      # Frames before classifying as "staying"

# Visualization
COLOR_MOVING = (0, 255, 0)    # Green
COLOR_STAYING = (0, 0, 255)   # Red
```

---

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Organize your training data:

```
training_workspace/dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # YOLO format labels
    └── val/
```

Label format (one line per object):
```
class_id center_x center_y width height
```
All values normalized to 0.0-1.0

### 2. Configure Training

Edit `training_workspace/data.yaml`:
```yaml
path: ./dataset
train: images/train
val: images/val

nc: 1  # Number of classes
names:
  0: cell
```

### 3. Run Training

```bash
cd training_workspace
python train_model.py
```

### 4. Deploy Model

```bash
cp runs/train/cell_detector/weights/best.pt ../assets/models/best.pt
```

---

## 📊 Module Documentation

### modules/detector.py
Wraps YOLO11 detection logic:
- `CellDetector`: Main detection class
- `detect()`: Returns list of detections
- `detect_and_annotate()`: Detects and draws bounding boxes

### modules/tracker.py
Handles movement tracking:
- `CellTracker`: Tracks cells across frames
- `update()`: Updates tracking with new detections
- `get_counts()`: Returns moving/staying/unknown counts

### modules/logger.py
Data logging functionality:
- `DataLogger`: Manages log files
- `log_counts()`: Records frame counts
- `log_movement()`: Records detailed movement
- `log_summary()`: Writes final summary
    * Implementing the "Virtual Death" algorithms based on user input.
    * Creating the Control Panel (Sliders for Poison/Food).
    * Managing the global state of the environment (e.g., Temperature dynamics).

### 👤 Module 3: The Analyst (Data & Prediction)
* **Focus:** Time-series Analysis and Future Forecasting.
* **Tech Stack:** `Scikit-Learn`, `PyQtGraph`, `NumPy`.
* **Responsibilities:**
    * Collecting real-time population data.
    * Training regression models on-the-fly to predict future trends.
    * Rendering live, dynamic line charts comparing "Actual" vs. "Predicted" growth.

---

## 💻 Tech Stack

* **Language:** Python 3.10+
* **User Interface:** PyQt6 (Qt for Python)
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Data Visualization:** PyQtGraph (for high-speed real-time plotting)
* **Machine Learning:** Scikit-Learn, NumPy

---

## 🎮 How to Use

1.  **Launch:** Run the application. The microscopic video feed will start automatically.
2.  **Observe:** Watch the **Green (Type A)** and **Red (Type B)** bounding boxes tracking the cells.
3.  **Analyze:** Look at the graph on the control panel. The dotted line represents the *predicted* future population.
4.  **Intervene:** * Drag the **"Toxin Level"** slider to the right.
    * Observe how the prediction graph dips.
    * Watch as "infected" cells on the screen get marked with a **X** symbol.

---

## 👥 The Team

Ali İhsan Sevindi
Elif Bozkurt
Sıla Bozkurt

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--
