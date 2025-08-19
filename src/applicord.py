
import sys
from PyQt5.QtWidgets import QApplication
from gui.windows import Windows, APP_NAME, APP_VERSION


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    main_window = Windows()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()