from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class Menu(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Menu label
        label = QLabel("Menu")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(label)
        
        # Add some spacing
        layout.addStretch()
        
        # To Entry button
        switch_entry_button = QPushButton("To Entry")
        switch_entry_button.clicked.connect(lambda: self.controller.show_frame("Entry"))
        switch_entry_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        layout.addWidget(switch_entry_button)
        
        self.setLayout(layout)
