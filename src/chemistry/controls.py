from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QGroupBox
from PyQt6.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    parameters_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        group = QGroupBox("Chemical Control")
        group_layout = QVBoxLayout()
        self.tox_label = QLabel("Toxicity: 0%")
        self.tox_slider = QSlider(Qt.Orientation.Horizontal)
        self.tox_slider.setRange(0, 100)
        self.tox_slider.valueChanged.connect(self.update_ui)
        group_layout.addWidget(self.tox_label)
        group_layout.addWidget(self.tox_slider)
        group.setLayout(group_layout)
        layout.addWidget(group)

    def update_ui(self):
        val = self.tox_slider.value()
        self.tox_label.setText(f"Toxicity: {val}%")
        self.parameters_changed.emit({"toxin": val})


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ControlPanel()
    window.show()
    sys.exit(app.exec())