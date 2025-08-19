from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QPushButton, QLabel, QMessageBox, QSpacerItem, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt
from data.data import DataToSheet
from datetime import datetime

class EntryPage(QWidget):
    LINKEDIN_URL = "https://www.linkedin.com/in/terryzhw/"
    GITHUB_URL = "https://github.com/terryzhw"
    
    MESSAGE_BOX_STYLE = """
        QMessageBox {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QMessageBox QPushButton {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 5px;
            padding: 8px;
            min-width: 60px;
        }
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Entry")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        profile_layout = QHBoxLayout()
        
        linkedin_btn = QPushButton(f"Copy LinkedIn\n{self.LINKEDIN_URL}")
        linkedin_btn.clicked.connect(self.copy_linkedin)
        linkedin_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                font-size: 11px;
                padding: 10px;
                margin: 5px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #24292e;
            }
        """)
        
        github_btn = QPushButton(f"Copy GitHub\n{self.GITHUB_URL}")
        github_btn.clicked.connect(self.copy_github)
        github_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                font-size: 11px;
                padding: 10px;
                margin: 5px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #24292e;
            }
        """)
        
        profile_layout.addWidget(linkedin_btn)
        profile_layout.addWidget(github_btn)
        layout.addLayout(profile_layout)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        company_label = QLabel("Company:")
        layout.addWidget(company_label)
        
        self.eCompany = QLineEdit()
        self.eCompany.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(self.eCompany)
        
        position_label = QLabel("Position:")
        layout.addWidget(position_label)
        
        self.ePosition = QLineEdit()
        self.ePosition.setStyleSheet("margin-bottom: 20px;")
        layout.addWidget(self.ePosition)
        
        self.add_button = QPushButton("Add Entry")
        self.add_button.clicked.connect(self.display_text)
        layout.addWidget(self.add_button)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
        
        self.eCompany.setFocus()

    def copy_linkedin(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.LINKEDIN_URL)

    def copy_github(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.GITHUB_URL)

    def display_text(self):
        data = DataToSheet()
        company = self.eCompany.text()
        position = self.ePosition.text()
        date = datetime.today().strftime('%m-%d-%Y')
        status = "Submitted"

        if company == "" or position == "":
            print("Error: Please fill in both fields")
        else:
            data.addData(company, position, date, status)
        
        self.eCompany.clear()
        self.ePosition.clear()
        self.eCompany.setFocus()