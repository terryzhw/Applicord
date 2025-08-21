from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from gui.menu_page import MenuPage
from gui.entry_page import EntryPage
from gui.classifier_page import ClassifierPage

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.gui_styles import MAIN_WINDOW_STYLE

class Windows(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Applicord")
        self.setGeometry(100, 100, 600, 400)
        
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.menu_frame = MenuPage(self)
        self.entry_frame = EntryPage(self)
        self.classifier_frame = ClassifierPage(self)

        self.stacked_widget.addWidget(self.menu_frame)
        self.stacked_widget.addWidget(self.entry_frame)
        self.stacked_widget.addWidget(self.classifier_frame)

        self.show_frame("Menu")

    def show_frame(self, frame_name):
        if frame_name == "Menu":
            self.stacked_widget.setCurrentWidget(self.menu_frame)
        elif frame_name == "Entry":
            self.stacked_widget.setCurrentWidget(self.entry_frame)
        elif frame_name == "Classifier":
            self.stacked_widget.setCurrentWidget(self.classifier_frame)

