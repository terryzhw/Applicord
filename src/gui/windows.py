from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from gui.menu_page import MenuPage
from gui.entry_page import EntryPage
from gui.classifier_page import ClassifierPage

DARK_THEME = {
    'background': '#2b2b2b',
    'text': '#ffffff',
    'button_bg': '#404040',
    'button_hover': '#505050',
    'button_pressed': '#303030',
    'border': '#555555',
    'input_bg': '#404040',
}

class Windows(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Applicord")
        self.setGeometry(100, 100, 600, 400)
        
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
            }}
            QWidget {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
            }}
            QPushButton {{
                background-color: {DARK_THEME['button_bg']};
                color: {DARK_THEME['text']};
                border: 1px solid {DARK_THEME['border']};
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DARK_THEME['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {DARK_THEME['button_pressed']};
            }}
            QLineEdit {{
                background-color: {DARK_THEME['input_bg']};
                color: {DARK_THEME['text']};
                border: 1px solid {DARK_THEME['border']};
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }}
            QLabel {{
                color: {DARK_THEME['text']};
                font-size: 14px;
            }}
        """)

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

