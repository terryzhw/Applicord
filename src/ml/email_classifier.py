import os
import pickle
from typing import Dict, Any
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

# Force CPU usage to avoid MPS-related instability issues on M1/M2 Macs
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class EmailClassifier:
    def __init__(self, model_name='roberta-base', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cpu')
    
    def prepare_datasets(self, csv_path: str, text_column: str, label_column: str):
        data_frame = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
        cleaned_data = self.clean_dataset(data_frame, text_column, label_column)
        
        email_texts = cleaned_data[text_column].fillna('').astype(str).tolist()
        email_labels = cleaned_data[label_column].tolist()
        encoded_labels = self.label_encoder.fit_transform(email_labels)
        
        return self.split_dataset(email_texts, encoded_labels)
    
    def clean_dataset(self, data_frame, text_column, label_column):
        # Only train on rejection vs non-rejection emails since that's our specific use case
        valid_labels = ['reject', 'not_reject']
        cleaned_data = data_frame[data_frame[label_column].isin(valid_labels)]
        cleaned_data = cleaned_data.dropna(subset=[text_column, label_column])
        
        print(f"Dataset size: {len(cleaned_data)} samples")
        print(f"Label distribution: {cleaned_data[label_column].value_counts().to_dict()}")
        
        return cleaned_data
    
    def split_dataset(self, email_texts, encoded_labels):
        # Reserve 20% for final testing to get unbiased performance metrics
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            email_texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Split remaining data into 60% train, 20% validation for hyperparameter tuning
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
        )
        
        print(f"Train set: {len(train_texts)} samples")
        print(f"Validation set: {len(val_texts)} samples")
        print(f"Test set: {len(test_texts)} samples")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    
    def create_model(self, num_classes: int = 2) -> RobertaForSequenceClassification:
        # Use RoBERTa because it performs better than BERT on classification tasks
        # Add dropout to prevent overfitting on small email datasets
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model.to(self.device)
        return self.model
    
    def train(self, csv_path: str, text_column: str = 'Email', 
              label_column: str = 'Status', epochs: int = 8, 
              batch_size: int = 16, learning_rate: float = 5e-5, 
              warmup_steps: int = 200, weight_decay: float = 0.01):
        dataset_splits = self.prepare_datasets(csv_path, text_column, label_column)
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = dataset_splits
        
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        num_classes = len(self.label_encoder.classes_)
        self.create_model(num_classes)
        
        class_weights = self.calculate_class_weights(train_labels)
        
        train_dataset = EmailDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = EmailDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        training_config = self.create_training_config(
            epochs, batch_size, learning_rate, warmup_steps, weight_decay
        )
        
        trainer = self.create_trainer(class_weights, training_config, train_dataset, val_dataset)
        trainer.train()
        
        return self.finalize_training(trainer)
    
    def calculate_class_weights(self, train_labels):
        # Balance classes since rejection emails are likely much rarer than regular emails
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Class weights: {class_weights.cpu()}")
        return class_weights
    
    def create_training_config(self, epochs, batch_size, learning_rate, warmup_steps, weight_decay):
        return TrainingArguments(
            output_dir="../model_temp",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=25,
            eval_strategy='steps',
            eval_steps=25,
            save_strategy='no',  # We'll save manually to control the final model
            load_best_model_at_end=False,
            max_grad_norm=1.0,
            lr_scheduler_type='cosine',
            report_to=[],
        )
    
    def create_trainer(self, class_weights, training_config, train_dataset, val_dataset):
        def compute_accuracy_metrics(eval_prediction):
            predictions, labels = eval_prediction
            predicted_labels = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predicted_labels)}
        
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Apply class weights to handle imbalanced dataset
                loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_function(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        return WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_accuracy_metrics
        )
    
    def finalize_training(self, trainer):
        validation_results = trainer.evaluate()
        print(f"Final Validation Loss: {validation_results['eval_loss']:.4f}")
        print(f"Final Validation Accuracy: {validation_results['eval_accuracy']:.4f}")
        
        model_save_path = "../model"
        print(f"Saving final model to {model_save_path}")
        self.save_model(model_save_path)
        
        self.cleanup_temp_files()
        
        return {
            'val_loss': validation_results['eval_loss'],
            'val_accuracy': validation_results['eval_accuracy']
        }
    
    def cleanup_temp_files(self):
        import shutil
        temp_directory = os.path.join("../model_temp")
        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)
            print(f"Cleaned up temporary training directory: {temp_directory}")
    
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
            predicted_idx = torch.argmax(outputs.logits, dim=1).item()
        
        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probabilities[0][predicted_idx].item()
        
        return {
            'is_rejection': predicted_class == 'reject',
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: probabilities[0][i].item() 
                for i, class_name in enumerate(self.label_encoder.classes_)
            }
        }
    
    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        with open(f"{filepath}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'test_texts': getattr(self, 'test_texts', None),
            'test_labels': getattr(self, 'test_labels', None)
        }
        with open(f"{filepath}/config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
    def load_model(self, filepath: str) -> None:
        if not os.path.isfile(f"{filepath}/config.pkl"):
            raise ValueError(f"No model found in {filepath}")
        
        print(f"Loading model from: {filepath}")
        with open(f"{filepath}/config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        self.test_texts = config.get('test_texts', None)
        self.test_labels = config.get('test_labels', None)
        
        self.model = RobertaForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = RobertaTokenizer.from_pretrained(filepath)
        self.model.to(self.device)
        
        with open(f"{filepath}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def evaluate_test_set(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        if self.test_texts is None or self.test_labels is None:
            raise ValueError("No test set available. Train the model first.")
        
        test_dataset = EmailDataset(
            self.test_texts, self.test_labels, self.tokenizer, self.max_length
        )
        trainer = Trainer(
            model=self.model,
            eval_dataset=test_dataset,
            compute_metrics=lambda eval_pred: {
                'accuracy': accuracy_score(
                    eval_pred[1],
                    np.argmax(eval_pred[0], axis=1)
                )
            }
        )
        
        test_results = trainer.evaluate()
        print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"Test Loss: {test_results['eval_loss']:.4f}")
        
        return {
            'test_accuracy': test_results['eval_accuracy'],
            'test_loss': test_results['eval_loss']
        }
    