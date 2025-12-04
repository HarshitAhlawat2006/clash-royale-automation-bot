# gui.py
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import sys

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        # Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # 4 image widgets + one number
        self.img_labels = [QLabel() for _ in range(4)]
        for i, lbl in enumerate(self.img_labels):
            self.layout.addWidget(lbl, i // 2, i % 2)

        self.number_label = QLabel("0")
        self.number_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        self.layout.addWidget(self.number_label, 2, 0, 1, 2)

    def set_images(self, images):
        """images: list of 4 numpy arrays (BGR from cv2)"""
        for lbl, img in zip(self.img_labels, images):
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w*c, QImage.Format.Format_RGB888)
            lbl.setPixmap(QPixmap.fromImage(qimg))

    def set_number(self, value):
        self.number_label.setText(str(value))


def start_gui():
    """Create and return the app + window so main.py can update it."""
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    return app, win