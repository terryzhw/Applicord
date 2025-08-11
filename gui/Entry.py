from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QPushButton, QLabel, QMessageBox, QSpacerItem, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QClipboard
from modules.data import DataToSheet
from datetime import datetime
import sys

class Entry(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        self.linkedin_url = "https://www.linkedin.com/in/terryzhw/"
        self.github_url = "https://github.com/terryzhw"
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Job Application Entry")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        profile_layout = QHBoxLayout()
        
        linkedin_btn = QPushButton(f"Copy LinkedIn\n{self.linkedin_url}")
        linkedin_btn.clicked.connect(self.copy_linkedin)
        linkedin_btn.setStyleSheet("""
            QPushButton {
                background-color: #0077b5;
                font-size: 11px;
                padding: 10px;
                margin: 5px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #005885;
            }
        """)
        
        github_btn = QPushButton(f"Copy GitHub\n{self.github_url}")
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
        self.eCompany.setPlaceholderText("Enter company name...")
        self.eCompany.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(self.eCompany)
        
        position_label = QLabel("Position:")
        layout.addWidget(position_label)
        
        self.ePosition = QLineEdit()
        self.ePosition.setPlaceholderText("Enter position title...")
        self.ePosition.setStyleSheet("margin-bottom: 20px;")
        layout.addWidget(self.ePosition)
        
        add_button = QPushButton("Add Entry")
        add_button.clicked.connect(self.display_text)
        add_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        layout.addWidget(add_button)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        layout.addWidget(back_button)
        
        self.setLayout(layout)
        
        self.eCompany.setFocus()

    def copy_linkedin(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.linkedin_url)
        self.show_message("LinkedIn URL copied to clipboard")

    def copy_github(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.github_url)
        self.show_message("GitHub URL copied to clipboard")

    def show_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Success")
        msg_box.setStyleSheet("""
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
        """)
        msg_box.exec_()

    def display_text(self):
        data = DataToSheet()
        company = self.eCompany.text()
        position = self.ePosition.text()
        date = datetime.today().strftime('%m-%d-%Y')
        status = "Submitted"

        if company == "" or position == "":
            self.show_error_message("Please fill in both fields")
        else:
            data.addData(company, position, date, status)
            self.show_message("Entry added")
        
        self.eCompany.clear()
        self.ePosition.clear()
        self.eCompany.setFocus()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStyleSheet("""
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
        """)
        msg_box.exec_()