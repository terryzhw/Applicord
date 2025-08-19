from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt


class ClassifierPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel("Classifier")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(label)
        
        layout.addStretch()

        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        back_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        
        layout.addWidget(back_button)
        
        self.setLayout(layout)