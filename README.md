# 🧬 Bio-Oracle: The Living Lab

![Python](https://img.shields.io/badge/Language-Python_3.10+-blue.svg?style=for-the-badge&logo=python)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg?style=for-the-badge&logo=qt)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-yellow.svg?style=for-the-badge&logo=ultralytics)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg?style=for-the-badge&logo=opencv)

<p align="center">
  <img src="docs/logo.png" alt="Bio-Oracle Logo" width="600">
  <br>
  <em>A Python-based Digital Twin engine merging Computer Vision, Machine Learning, and Gamification.</em>
</p>

---

## 📖 Overview

**Bio-Oracle** is a real-time bio-simulation platform that transforms passive microscopic video feeds into an interactive laboratory environment.

Built with **Python** and **PyQt6**, the system uses **YOLOv8** to track living cells, **Scikit-Learn** to predict population trends, and provides a gamified interface where users can virtually intervene (e.g., injecting toxins or nutrients) to alter the biological outcome.

This project implements a **Vertical Slice Architecture**, ensuring a modular and scalable codebase where Computer Vision, Backend Logic, and UI components are integrated within functional domains.

---

## 🚀 Key Features

* **🦠 Synthetic Vision:** Real-time object detection and tracking of cells using YOLOv8, rendered directly onto the video feed via PyQt painters.
* **🧪 Interactive Simulation:** A physics-based game engine where users control environmental variables (Toxicity, pH, Temperature) to manipulate cell survival rates.
* **📈 Predictive Analytics:** Live forecasting of population growth using Linear/Polynomial Regression, visualized with high-performance **PyQtGraph** charts.
* **🎮 Gamified HUD:** A modern "Heads-Up Display" interface overlaying the biological footage.

---

## 🏗️ Vertical Slice Architecture & Roles

The project is divided into three distinct modules. Each team member owns the **Full Stack** (UI + Backend + AI) of their respective domain:

### 👤 Module 1: The Biologist (Vision & Entities)
* **Focus:** Detection, Tracking, and Visualization of life forms.
* **Tech Stack:** `OpenCV`, `YOLOv8`, `QPainter`.
* **Responsibilities:**
    * Handling video ingestion and frame processing.
    * Running AI inference to detect cells (Type A vs. Type B).
    * Rendering bounding boxes and unique IDs over the video feed.

### 👤 Module 2: The Chemist (Environment & Controls)
* **Focus:** Simulation Logic and User Interaction.
* **Tech Stack:** `PyQt6 Widgets`, `Simulation Math`.
* **Responsibilities:**
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
