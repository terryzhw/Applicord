#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import os
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EmailClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_pipeline = None
        self.sentiment_analyzer = None
        
        try:
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                except:
                    pass
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
            print("Warning: NLTK sentiment analysis unavailable, proceeding without it")
        
        self.rejection_keywords = [
            'unfortunately', 'regret', 'decided not', 'not selected', 'not moving forward',
            'move forward with other', 'pursue other candidates', 'other candidates',
            'better match', 'not proceed', 'will not be moving', 'unable to offer',
            'not be able to offer', 'decided to move forward', 'impressed with your background',
            'have decided not to', 'decided to pursue', 'not the right fit', 'at this time',
            'we received many', 'many qualified', 'another candidate', 'better aligned',
            'not a match', 'filled this position', 'position has been filled', 'thank you for applying',
            'we appreciate your interest', 'we will keep your resume', 'keep you in mind',
            'future opportunities', 'not successful', 'wish you well', 'best of luck',
            'cannot offer', 'we have chosen', 'competitive pool', 'difficult decision',
            'many strong candidates', 'after careful consideration', 'will not be considered'
        ]
        
        self.positive_keywords = [
            'interview', 'schedule', 'next steps', 'congratulations', 'offer',
            'pleased to inform', 'happy to inform', 'excited to', 'would like to',
            'looking forward', 'interested in speaking', 'phone call', 'discussion',
            'schedule a time', 'available for', 'move to the next', 'second round',
            'final round', 'onsite', 'video call', 'zoom', 'teams meeting'
        ]
    
    def extract_advanced_features(self, text):
        if pd.isna(text):
            return pd.Series([0] * 50)
            
        text_str = str(text)
        text_lower = text_str.lower()
        
        features = {}
        
        rejection_weights = {
            'unfortunately': 3, 'regret': 3, 'decided not': 3, 'not selected': 3,
            'not moving forward': 3, 'other candidates': 2, 'better match': 2,
            'not proceed': 3, 'unable to offer': 3, 'not the right fit': 2,
            'at this time': 1, 'many qualified': 2, 'another candidate': 2,
            'filled this position': 3, 'position has been filled': 3,
            'thank you for applying': 1, 'we appreciate your interest': 1,
            'future opportunities': 2, 'wish you well': 2, 'best of luck': 2
        }
        
        positive_weights = {
            'interview': 3, 'schedule': 3, 'next steps': 3, 'congratulations': 3,
            'offer': 3, 'excited to': 2, 'looking forward': 2, 'phone call': 2,
            'schedule a time': 3, 'move to the next': 3, 'second round': 3,
            'final round': 3, 'video call': 2
        }
        
        rejection_score = 0
        positive_score = 0
        
        for keyword, weight in rejection_weights.items():
            if keyword in text_lower:
                features[f'reject_{keyword.replace(" ", "_")}'] = weight
                rejection_score += weight
            else:
                features[f'reject_{keyword.replace(" ", "_")}'] = 0
                
        for keyword, weight in positive_weights.items():
            if keyword in text_lower:
                features[f'positive_{keyword.replace(" ", "_")}'] = weight
                positive_score += weight
            else:
                features[f'positive_{keyword.replace(" ", "_")}'] = 0
        
        features['text_length'] = len(text_str)
        features['word_count'] = len(text_str.split())
        features['sentence_count'] = len([s for s in text_str.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in text_str.split()]) if text_str.split() else 0
        features['rejection_score'] = rejection_score
        features['positive_score'] = positive_score
        features['score_ratio'] = rejection_score / (positive_score + 1)
        
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer.polarity_scores(text_str)
                features['sentiment_compound'] = sentiment['compound']
                features['sentiment_positive'] = sentiment['pos']
                features['sentiment_negative'] = sentiment['neg']
                features['sentiment_neutral'] = sentiment['neu']
            except:
                features['sentiment_compound'] = 0
                features['sentiment_positive'] = 0
                features['sentiment_negative'] = 0
                features['sentiment_neutral'] = 1
        else:
            features['sentiment_compound'] = -0.5 if rejection_score > positive_score else 0.2
            features['sentiment_positive'] = 0.1 if positive_score > 0 else 0
            features['sentiment_negative'] = 0.6 if rejection_score > 0 else 0.1
            features['sentiment_neutral'] = 1 - (features['sentiment_positive'] + features['sentiment_negative'])
        
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text_str)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text_str)
        except:
            features['flesch_reading_ease'] = 50
            features['flesch_kincaid_grade'] = 8
        
        features['exclamation_count'] = text_str.count('!')
        features['question_count'] = text_str.count('?')
        features['uppercase_ratio'] = sum(1 for c in text_str if c.isupper()) / len(text_str) if text_str else 0
        features['punctuation_density'] = sum(1 for c in text_str if c in '.,!?;:') / len(text_str) if text_str else 0
        
        features['has_greeting'] = 1 if any(greeting in text_lower for greeting in ['dear', 'hello', 'hi']) else 0
        features['has_signature'] = 1 if any(sig in text_lower for sig in ['best regards', 'sincerely', 'best wishes']) else 0
        features['formal_tone'] = 1 if any(formal in text_lower for formal in ['we regret', 'we appreciate', 'thank you for your time']) else 0
        
        return pd.Series(features)
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\!\?\,]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def load_data(self, csv_path='data.csv'):
        df = pd.read_csv(csv_path)
        
        df['Email'] = df['Email'].fillna('')
        df['Status'] = df['Status'].fillna('unknown')
        
        advanced_features = df['Email'].apply(self.extract_advanced_features)
        
        df['cleaned_email'] = df['Email'].apply(self.preprocess_text)
        
        return df, advanced_features
    
    def train_model(self, csv_path='data.csv'):
        print("Loading and preprocessing data...")
        df, advanced_features = self.load_data(csv_path)
        
        X_text = df['cleaned_email']
        X_features = advanced_features
        y = df['Status']
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset loaded: {len(df)} emails")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Class distribution:")
        print(df['Status'].value_counts())
        
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            X_text, X_features, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )
        
        vectorizers = {
            'tfidf_word': TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
                stop_words='english',
                sublinear_tf=True,
                analyzer='word'
            ),
            'tfidf_char': TfidfVectorizer(
                max_features=3000,
                ngram_range=(3, 5),
                min_df=1,
                max_df=0.95,
                analyzer='char_wb'
            ),
            'count': CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9,
                stop_words='english'
            )
        }
        
        print("Creating text representations...")
        text_features_train = []
        text_features_test = []
        
        for name, vectorizer in vectorizers.items():
            print(f"  Processing {name} vectorizer...")
            train_vec = vectorizer.fit_transform(X_text_train)
            test_vec = vectorizer.transform(X_text_test)
            text_features_train.append(train_vec)
            text_features_test.append(test_vec)
            
        self.vectorizer = vectorizers['tfidf_word']
        
        from scipy.sparse import hstack, csr_matrix
        X_text_combined_train = hstack(text_features_train)
        X_text_combined_test = hstack(text_features_test)
        
        scaler = MinMaxScaler()
        X_feat_train_scaled = scaler.fit_transform(X_feat_train.fillna(0))
        X_feat_test_scaled = scaler.transform(X_feat_test.fillna(0))
        
        X_train_final = hstack([X_text_combined_train, csr_matrix(X_feat_train_scaled)])
        X_test_final = hstack([X_text_combined_test, csr_matrix(X_feat_test_scaled)])
        
        print(f"Final feature matrix shape: {X_train_final.shape}")
        
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [800, 1000, 1200],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [500, 800, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [6, 8, 10, 12],
                    'min_samples_split': [2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [800, 1000, 1200],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [2, 3],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced']
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(200, 100), (300, 150), (400, 200, 100)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01],
                    'solver': ['adam'],
                    'activation': ['relu', 'tanh']
                }
            }
        }
        
        best_models = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\\nStarting intensive hyperparameter optimization (this will take 5+ minutes)...")
        
        for name, config in models.items():
            print(f"\\nOptimizing {name}...")
            
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print(f"  Fitting {len(config['params'])} parameter combinations with 5-fold CV...")
            grid_search.fit(X_train_final, y_train)
            
            best_models[name] = {
                'model': grid_search.best_estimator_,
                'score': grid_search.best_score_,
                'params': grid_search.best_params_
            }
            
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            print(f"  Best params: {grid_search.best_params_}")
        
        print("\\nCreating ensemble model...")
        ensemble_models = []
        for name, info in best_models.items():
            ensemble_models.append((name, info['model']))
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        print("Training final ensemble...")
        ensemble.fit(X_train_final, y_train)
        
        y_pred_ensemble = ensemble.predict(X_test_final)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"\\nEnsemble accuracy: {ensemble_accuracy:.4f}")
        
        best_individual = max(best_models.items(), key=lambda x: x[1]['score'])
        individual_pred = best_individual[1]['model'].predict(X_test_final)
        individual_accuracy = accuracy_score(y_test, individual_pred)
        
        print(f"Best individual model ({best_individual[0]}): {individual_accuracy:.4f}")
        
        if ensemble_accuracy > individual_accuracy:
            self.model = ensemble
            final_accuracy = ensemble_accuracy
            final_name = "Ensemble"
        else:
            self.model = best_individual[1]['model']
            final_accuracy = individual_accuracy
            final_name = best_individual[0]
        
        print(f"\\nFinal model: {final_name} with accuracy: {final_accuracy:.4f}")
        
        y_pred_final = self.model.predict(X_test_final)
        print(f"\\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_final, target_names=self.label_encoder.classes_))
        print(f"\\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_final))
        
        self.scaler = scaler
        self.vectorizers = vectorizers
        
        self.save_model()
        
        return final_accuracy
    
    def predict(self, email_text):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained or loaded. Please train the model first.")
        
        advanced_features = self.extract_advanced_features(email_text)
        
        cleaned_text = self.preprocess_text(email_text)
        
        if hasattr(self, 'vectorizers') and self.vectorizers:
            text_features = []
            for name, vectorizer in self.vectorizers.items():
                vec_result = vectorizer.transform([cleaned_text])
                text_features.append(vec_result)
            
            from scipy.sparse import hstack, csr_matrix
            text_combined = hstack(text_features)
        else:
            text_combined = self.vectorizer.transform([cleaned_text])
        
        if hasattr(self, 'scaler') and self.scaler:
            advanced_features_scaled = self.scaler.transform(advanced_features.values.reshape(1, -1))
        else:
            advanced_features_scaled = advanced_features.values.reshape(1, -1)
        
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([text_combined, csr_matrix(advanced_features_scaled)])
        
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {
            cls: prob for cls, prob in zip(self.label_encoder.classes_, probability)
        }
        
        return predicted_class, prob_dict
    
    def save_model(self, model_dir='models'):
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'email_classifier_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'email_vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'email_label_encoder.pkl'))
        
        if hasattr(self, 'vectorizers') and self.vectorizers:
            joblib.dump(self.vectorizers, os.path.join(model_dir, 'email_vectorizers.pkl'))
        
        if hasattr(self, 'scaler') and self.scaler:
            joblib.dump(self.scaler, os.path.join(model_dir, 'email_scaler.pkl'))
        
        print(f"Enhanced model saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        try:
            self.model = joblib.load(os.path.join(model_dir, 'email_classifier_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'email_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'email_label_encoder.pkl'))
            
            try:
                self.vectorizers = joblib.load(os.path.join(model_dir, 'email_vectorizers.pkl'))
            except FileNotFoundError:
                pass
            
            try:
                self.scaler = joblib.load(os.path.join(model_dir, 'email_scaler.pkl'))
            except FileNotFoundError:
                pass
            
            print("Enhanced model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No pre-trained model found. Please train a model first.")
            return False

if __name__ == "__main__":
    classifier = EmailClassifier()
    
    accuracy = classifier.train_model('data.csv')
    print(f"\\nFinal model accuracy: {accuracy:.4f}")
    
    test_email = """Thank you for your interest in our company. 
                    Unfortunately, we have decided to move forward with other candidates."""
    
    prediction, probabilities = classifier.predict(test_email)
    print(f"\\nTest prediction: {prediction}")
    print(f"Probabilities: {probabilities}")