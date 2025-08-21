from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
import os
from ml.email_classifier import EmailClassifier
from search.email_search import CompanySearcher

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.gui_styles import COMMON_STYLES


class ClassifierPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel("Classifier")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(COMMON_STYLES['title'])
        layout.addWidget(label)
        
        layout.addStretch()


        classifier_label = QLabel("Epochs:")
        layout.addWidget(classifier_label)

        self.eClassifier = QLineEdit()
        self.eClassifier.setPlaceholderText("Enter an integer...")
        self.eClassifier.setStyleSheet(COMMON_STYLES['input_margin_small'])
        layout.addWidget(self.eClassifier)


        self.classifier_button = QPushButton("Train Model")
        self.classifier_button.clicked.connect(self.run_trainer)
        layout.addWidget(self.classifier_button)


        self.search_button = QPushButton("Run Search")
        self.search_button.clicked.connect(self.run_search)
        layout.addWidget(self.search_button)




        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
    
    def run_trainer(self):
        epochs_input = self.eClassifier.text()
        
        if not self.validate_training_input(epochs_input):
            return
        
        email_classifier = EmailClassifier()
        training_epochs = int(epochs_input)
        email_classifier.train("../data.csv", epochs=training_epochs)
    
    def validate_training_input(self, epochs_input):
        # Prevent accidental retraining which would overwrite existing model
        model_path = "../../model"
        if os.path.exists(model_path) and os.path.isfile(f"{model_path}/config.pkl"):
            print("Error: model already trained")
            return False
        
        if epochs_input == "":
            print("Error: Please fill in field")
            return False
        
        # Ensure epochs is a valid positive integer
        if not epochs_input.isdigit():
            print("Error: Please enter an integer")
            return False
        
        return True

    def run_search(self):
        email_searcher = CompanySearcher()
        email_searcher.search_all_companies_from_spreadsheet()


