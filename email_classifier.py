import os
import pickle
from typing import Dict, Tuple, List, Any

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmailClassifier:
    def __init__(self, model_name='roberta-large', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cpu')
    
    def load_data(self, csv_path: str, text_column: str, label_column: str) -> Tuple[List[str], List[str], List[int], List[int]]:
        df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
        
        valid_labels = ['reject', 'not_reject']
        df = df[df[label_column].isin(valid_labels)].dropna(subset=[text_column, label_column])
        
        
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[label_column].tolist()
        y = self.label_encoder.fit_transform(labels)
        return train_test_split(texts, y, test_size=0.2, random_state=42)
    
    def create_model(self, num_labels: int = 2) -> RobertaForSequenceClassification:
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.model.to(self.device)
        return self.model
    
    def train(self, csv_path: str, text_column: str = 'email_text', label_column: str = 'is_rejection', 
              epochs: int = 3, batch_size: int = 2, learning_rate: float = 2e-5, warmup_steps: int = 500, weight_decay: float = 0.01) -> float:
        X_train, X_test, y_train, y_test = self.load_data(csv_path, text_column, label_column)
        
        
        num_labels = len(self.label_encoder.classes_)
        self.create_model(num_labels)
        
        train_dataset = EmailDataset(X_train, y_train, self.tokenizer, self.max_length)
        test_dataset = EmailDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        training_args = TrainingArguments(
            output_dir='./model',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            greater_is_better=True,
            dataloader_pin_memory=False,
        )
        
        def compute_metrics(eval_pred) -> Dict[str, float]:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        print(f"Final Test Accuracy: {eval_results['eval_accuracy']:.4f}")
        
        return eval_results['eval_accuracy']
    
    def predict(self, email_text: str) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        inputs = self.tokenizer(
            email_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted = torch.argmax(outputs.logits, dim=1)
        
        predicted_label = predicted.item()
        confidence = probabilities[0][predicted_label].item()
        predicted_class = self.label_encoder.inverse_transform([predicted_label])[0]
        is_rejection = (predicted_class == 'reject')
        
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[0][i].item()
        
        return {
            'is_rejection': is_rejection,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        with open(f"{filepath}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        config = {'model_name': self.model_name, 'max_length': self.max_length}
        with open(f"{filepath}/config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
    
    def load_model(self, filepath: str) -> None:
        with open(f"{filepath}/config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        
        self.model = RobertaForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = RobertaTokenizer.from_pretrained(filepath)
        self.model.to(self.device)
        
        with open(f"{filepath}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        

if __name__ == "__main__":
    classifier = EmailClassifier(model_name='roberta-large')
    model_path = './model'
    
    if os.path.exists(model_path) and os.path.isfile(f"{model_path}/config.pkl"):
        classifier.load_model(model_path)
        print("Model loaded successfully")
    else:
        accuracy = classifier.train('data.csv', text_column='Email', label_column='Status', epochs=10)
        classifier.save_model(model_path)
        print(f"Accuracy: {accuracy:.4f}")
    test_email = "Dear Terrance, After further consideration of your resume, we are moving on with another candidate. We wish you good luck on your future endeavors"
    result = classifier.predict(test_email)
    print(result)
