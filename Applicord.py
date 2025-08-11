import sys
from PyQt5.QtWidgets import QApplication
from gui.windows import Windows

if __name__ == "__main__":
    app = QApplication(sys.argv)
    interface = Windows()
    interface.show()
    sys.exit(app.exec_())