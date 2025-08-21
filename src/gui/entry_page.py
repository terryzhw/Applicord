from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QPushButton, QLabel, QMessageBox, QSpacerItem, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt
from sheet.sheet_manager import SheetManager
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.gui_styles import COMMON_STYLES, get_special_button_style

class EntryPage(QWidget):
    # Personal URLs for quick access during job applications
    LINKEDIN_URL = "https://www.linkedin.com/in/terryzhw/"
    GITHUB_URL = "https://github.com/terryzhw"
    

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Entry")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(COMMON_STYLES['title'])
        layout.addWidget(title)
        
        profile_layout = QHBoxLayout()
        
        linkedin_btn = QPushButton(f"Copy LinkedIn\n{self.LINKEDIN_URL}")
        linkedin_btn.clicked.connect(self.copy_linkedin)
        linkedin_btn.setStyleSheet(get_special_button_style())
        
        github_btn = QPushButton(f"Copy GitHub\n{self.GITHUB_URL}")
        github_btn.clicked.connect(self.copy_github)
        github_btn.setStyleSheet(get_special_button_style())
        
        profile_layout.addWidget(linkedin_btn)
        profile_layout.addWidget(github_btn)
        layout.addLayout(profile_layout)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        company_label = QLabel("Company:")
        layout.addWidget(company_label)
        
        self.eCompany = QLineEdit()
        self.eCompany.setStyleSheet(COMMON_STYLES['input_margin_small'])
        layout.addWidget(self.eCompany)
        
        position_label = QLabel("Position:")
        layout.addWidget(position_label)
        
        self.ePosition = QLineEdit()
        self.ePosition.setStyleSheet(COMMON_STYLES['input_margin_large'])
        layout.addWidget(self.ePosition)
        
        self.add_button = QPushButton("Add Entry")
        self.add_button.clicked.connect(self.display_text)
        layout.addWidget(self.add_button)
        
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
        
        # Set focus to company field for faster data entry
        self.eCompany.setFocus()

    def copy_linkedin(self):
        system_clipboard = QApplication.clipboard()
        system_clipboard.setText(self.LINKEDIN_URL)

    def copy_github(self):
        system_clipboard = QApplication.clipboard()
        system_clipboard.setText(self.GITHUB_URL)

    def display_text(self):
        company_name = self.eCompany.text()
        position_title = self.ePosition.text()
        
        if not self.validate_input(company_name, position_title):
            return
        
        # Add to spreadsheet for tracking application progress
        self.add_entry_to_sheet(company_name, position_title)
        
        self.reset_form()
    
    def validate_input(self, company_name, position_title):
        # Ensure both fields are filled to maintain data quality
        if company_name == "" or position_title == "":
            print("Error: Please fill in both fields")
            return False
        return True
    
    def add_entry_to_sheet(self, company_name, position_title):
        sheet_manager = SheetManager()
        current_date = datetime.today().strftime('%m-%d-%Y')
        initial_status = "Submitted"
        
        sheet_manager.addData(company_name, position_title, current_date, initial_status)
    
    def reset_form(self):
        self.eCompany.clear()
        self.ePosition.clear()
        # Return focus to company field for next entry
        self.eCompany.setFocus()