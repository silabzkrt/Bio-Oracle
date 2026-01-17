# 🧬 Bio-Oracle: The Living Lab

![C++](https://img.shields.io/badge/Language-C++17-blue.svg?style=for-the-badge&logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV_4.x-green.svg?style=for-the-badge&logo=opencv)
![ImGui](https://img.shields.io/badge/UI-Dear_ImGui-red.svg?style=for-the-badge)
![AI](https://img.shields.io/badge/Model-YOLOv8-yellow.svg?style=for-the-badge)

<p align="center">
  <img src="docs/logo.png" alt="Bio-Oracle Logo" width="600">
  <br>
  <em>Bridging the gap between biological observation and digital intervention.</em>
</p>

---

## 📖 Overview

**Bio-Oracle** is a high-performance **Digital Twin** bio-simulation engine built with C++. It transforms passive microscopic video feeds into an interactive, gamified laboratory environment.

Using **Computer Vision** (YOLOv8 & OpenCV), the system tracks living cells in real-time. Using **Machine Learning**, it predicts future population trends. Through an **Augmented Reality (AR)** interface (Dear ImGui), it allows users to virtually manipulate the environment—injecting digital "toxins" or "nutrients" to alter the biological outcome dynamically.

This project was developed as a Computer Science capstone project at **Bilkent University**.

---

## 🚀 Key Features

### 👁️ The Eye (Perception Layer)
* **Real-time Detection:** Utilizes **YOLOv8** (ONNX Runtime) to detect and classify cell types (Type A vs. Type B) from raw video feeds.
* **Object Tracking:** Assigns unique IDs to cells to track their lifespan and movement across frames.
* **Segmentation Overlay:** Renders visual masks over organic matter for immediate visual feedback.

### 🧠 The Brain (Simulation & Prediction Layer)
* **Predictive Modeling:** Implements time-series regression algorithms to forecast population growth/decay 5-10 minutes into the future.
* **Digital Twin Logic:** Manages the state of "Virtual Cells." It can simulate the death of a cell based on user input, even if the cell is alive in the source video.
* **Dynamic Parameters:** Calculates the impact of environmental variables (Temperature, pH, Antibiotics) on biological behavior.

### 🎮 The Interface (Interaction Layer)
* **Heads-Up Display (HUD):** A responsive overlay built with **Dear ImGui** that sits on top of the video feed.
* **Live Analytics:** Uses **ImPlot** to render real-time graphs showing the "Predicted vs. Actual" population trends.
* **Gamified Controls:** Users can interact with sliders to change the simulation state (e.g., *Increase Toxicity to 80%*).

---

## 🛠️ System Architecture

The project follows a **Monolithic Vertical Slice** architecture, optimized for low-latency C++ performance.

| Module | Responsibility | Tech Stack |
| :--- | :--- | :--- |
| **Vision System** | Video Ingestion, Pre-processing, Inference (YOLO) | OpenCV DNN, ONNX |
| **Game Engine** | Physics calculation, Growth Logic, ML Prediction | Standard C++ Library |
| **GUI Renderer** | Window management, Graph plotting, User Input | Dear ImGui, OpenGL/DirectX |

---

## 💻 Tech Stack & Dependencies

* **Language:** C++17 or C++20
* **IDE:** Visual Studio 2022 (Recommended)
* **Computer Vision:** [OpenCV 4.x](https://opencv.org/)
* **GUI:** [Dear ImGui](https://github.com/ocornut/imgui) + [ImPlot](https://github.com/epezent/implot)
* **AI Model:** [YOLOv8](https://github.com/ultralytics/ultralytics) (Exported as `.onnx`)

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

* **[İsim 1]:** *Perception Engineer* - Computer Vision & Object Detection.
* **[İsim 2]:** *Simulation Architect* - ML Backend & Game Logic.
* **[İsim 3]:** *Interface Developer* - UI/UX & Data Visualization.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--
