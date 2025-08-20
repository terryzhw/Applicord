
import sys
from PyQt5.QtWidgets import QApplication
from gui.windows import Windows


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Applicord")
    main_window = Windows()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()