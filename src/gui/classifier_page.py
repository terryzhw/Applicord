from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
import os
from ml.email_classifier import EmailClassifier
from search.email_search import CompanySearcher


class ClassifierPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel("Classifier")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(label)
        
        layout.addStretch()

        # Classifier area

        classifier_label = QLabel("Epochs:")
        layout.addWidget(classifier_label)

        self.eClassifier = QLineEdit()
        self.eClassifier.setPlaceholderText("Enter an integer...")
        self.eClassifier.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(self.eClassifier)


        self.classifier_button = QPushButton("Train Model")
        self.classifier_button.clicked.connect(self.run_trainer)
        layout.addWidget(self.classifier_button)

        # Search area

        self.search_button = QPushButton("Run Search")
        self.search_button.clicked.connect(self.run_search)
        layout.addWidget(self.search_button)




        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
    
    def run_trainer(self):
        classifier = EmailClassifier()
        epoch = self.eClassifier.text()

        if os.path.exists("../../model") and os.path.isfile(f"../../model/config.pkl"):
            print("Error: model already trained")
        elif epoch == "":
            print("Error: Please fill in field")
        elif not epoch.isdigit():
            print("Error: Please enter an integer")
        else:
            classifier.train("../data.csv", epochs=int(epoch))
            classifier.save_model("../../model")

    def run_search(self):
        searcher = CompanySearcher()
        searcher.search_all_companies_from_spreadsheet()


