import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from gui.menu import Menu
from gui.entry import Entry


class Windows(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Applicord")
        self.setGeometry(100, 100, 600, 400)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QLineEdit {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.menu_frame = Menu(self)
        self.entry_frame = Entry(self)

        self.stacked_widget.addWidget(self.menu_frame)
        self.stacked_widget.addWidget(self.entry_frame)

        self.show_frame("Menu")

    def show_frame(self, frame_name):
        if frame_name == "Menu":
            self.stacked_widget.setCurrentWidget(self.menu_frame)
        elif frame_name == "Entry":
            self.stacked_widget.setCurrentWidget(self.entry_frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Windows()
    window.show()
    sys.exit(app.exec_())