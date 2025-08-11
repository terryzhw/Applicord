from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QMessageBox, QProgressBar, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.email_classifier import EmailClassifier

class TrainingThread(QThread):
    finished = pyqtSignal(float)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            classifier = EmailClassifier()
            accuracy = classifier.train_model('data.csv')
            self.finished.emit(accuracy)
        except Exception as e:
            self.error.emit(str(e))

class ClassifierPage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.classifier = EmailClassifier()
        self.classifier.load_model()
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        label = QLabel("Model Training & Email Prediction")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        train_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        layout.addWidget(train_button)
        
        email_label = QLabel("Email Prediction Test:")
        email_label.setStyleSheet("font-size: 14px; margin-top: 20px;")
        layout.addWidget(email_label)
        
        self.email_input = QTextEdit()
        self.email_input.setMaximumHeight(150)
        self.email_input.setStyleSheet("""
            QTextEdit {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.email_input)
        
        predict_layout = QHBoxLayout()
        
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict_email)
        predict_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 5px;
            }
        """)
        predict_layout.addWidget(predict_button)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_input)
        clear_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 5px;
            }
        """)
        predict_layout.addWidget(clear_button)
        
        layout.addLayout(predict_layout)
        
        layout.addStretch()
        
        back_button = QPushButton("Back to Menu")
        back_button.clicked.connect(lambda: self.controller.show_frame("Menu"))
        back_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 12px;
                margin: 10px;
            }
        """)
        layout.addWidget(back_button)
        
        self.setLayout(layout)
    
    def train_model(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training in Progress", "Model training is already in progress.")
            return
        
        reply = QMessageBox.question(self, 'Train Model', 
                                   'This will train a new model on the data.csv file.\nThis may take a few minutes. Continue?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            self.training_thread = TrainingThread()
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            self.training_thread.start()
    
    def on_training_finished(self, accuracy):
        self.progress_bar.setVisible(False)
        self.classifier.load_model()
        QMessageBox.information(self, "Training Complete", 
                              f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
    
    def on_training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Training Error", f"Error during training:\n{error_msg}")
    
    def predict_email(self):
        email_text = self.email_input.toPlainText().strip()
        
        if not email_text:
            QMessageBox.warning(self, "No Input", "Please enter email text to classify.")
            return
        
        if not self.classifier.model:
            reply = QMessageBox.question(self, 'No Model Found', 
                                       'No trained model found. Would you like to train a model first?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.train_model()
            return
        
        try:
            prediction, probabilities = self.classifier.predict(email_text)
            
            result_text = f"Prediction: {prediction.upper()}\n\n"
            result_text += "Probabilities:\n"
            for class_name, prob in probabilities.items():
                result_text += f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)\n"
            
            if prediction == 'reject':
                QMessageBox.warning(self, "Classification Result", result_text)
            else:
                QMessageBox.information(self, "Classification Result", result_text)
                
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error during prediction:\n{str(e)}")
    
    def clear_input(self):
        self.email_input.clear()