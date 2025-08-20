from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.gui_styles import COMMON_STYLES


class MenuPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel("Menu")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(COMMON_STYLES['title'])
        layout.addWidget(label)
        
        layout.addStretch()
        
        switch_entry_button = QPushButton("To Entry")
        switch_entry_button.clicked.connect(
            lambda: self.controller.show_frame("Entry")
        )
        switch_entry_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        
        switch_classifier_button = QPushButton("To Classifier")
        switch_classifier_button.clicked.connect(
            lambda: self.controller.show_frame("Classifier")
        )
        switch_classifier_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        layout.addWidget(switch_entry_button)
        layout.addWidget(switch_classifier_button) 
        self.setLayout(layout)
