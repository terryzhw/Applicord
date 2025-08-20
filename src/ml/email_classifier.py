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
        df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
        
        valid_labels = ['reject', 'not_reject']
        df = df[df[label_column].isin(valid_labels)].dropna(subset=[text_column, label_column])
        
        print(f"Dataset size: {len(df)} samples")
        print(f"Label distribution: {df[label_column].value_counts().to_dict()}")
        
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[label_column].tolist()
        y = self.label_encoder.fit_transform(labels)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self, num_classes: int = 2) -> RobertaForSequenceClassification:
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
        data = self.prepare_datasets(csv_path, text_column, label_column)
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = data
        
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        num_classes = len(self.label_encoder.classes_)
        self.create_model(num_classes)
        
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Class weights: {class_weights.cpu()}")
        
        train_dataset = EmailDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = EmailDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        training_args = TrainingArguments(
            output_dir='../model',
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
            save_strategy='steps',
            save_steps=25,
            max_grad_norm=1.0,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            lr_scheduler_type='cosine',
            report_to=[],
        )
        
        def compute_metrics(eval_pred) -> Dict[str, float]:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=3, 
                early_stopping_threshold=0.01
            )]
        )
        
        trainer.train()
        
        val_results = trainer.evaluate()
        print(f"Final Validation Loss: {val_results['eval_loss']:.4f}")
        print(f"Final Validation Accuracy: {val_results['eval_accuracy']:.4f}")
        
        return {
            'val_loss': val_results['eval_loss'],
            'val_accuracy': val_results['eval_accuracy']
        }
    
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
        # Check if complete model exists
        if os.path.isfile(f"{filepath}/config.pkl"):
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
        else:
            # Load from latest checkpoint
            checkpoint_dirs = [d for d in os.listdir(filepath) if d.startswith('checkpoint-')]
            if not checkpoint_dirs:
                raise ValueError(f"No model or checkpoints found in {filepath}")
            
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(filepath, latest_checkpoint)
            
            print(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Use default values when loading from checkpoint
            self.model_name = 'roberta-base'  # Default value
            self.max_length = 512  # Default value
            self.test_texts = None
            self.test_labels = None
            
            self.model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Create a default label encoder for reject/not_reject classes
            self.label_encoder.fit(['not_reject', 'reject'])
    
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
    