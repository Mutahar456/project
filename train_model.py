"""
Optimized XGBoost Model for Job Department Classification
Target: 90%+ accuracy on Department Classification task
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class JobDepartmentClassifier:
    def __init__(self, max_features=5000, test_size=0.15, val_size=0.15):
        self.max_features = max_features
        self.test_size = test_size
        self.val_size = val_size
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_features(self, df):
        """Extract features from job postings"""
        features = []
        
        # Text features
        df['text_clean'] = df['text'].apply(self.clean_text)
        df['text_length'] = df['text_clean'].str.len()
        df['word_count'] = df['text_clean'].str.split().str.len()
        df['sentence_count'] = df['text_clean'].str.count(r'[.!?]+')
        
        # Title features
        df['title_clean'] = df['name'].apply(self.clean_text)
        df['title_length'] = df['title_clean'].str.len()
        df['title_word_count'] = df['title_clean'].str.split().str.len()
        
        # Source encoding
        df['source_encoded'] = pd.Categorical(df['source']).codes
        
        # Company name features (if available)
        df['has_company'] = df['orgCompany'].apply(lambda x: 1 if isinstance(x, dict) and x.get('name') else 0)
        
        # Location features
        df['has_location'] = df['orgAddress'].apply(lambda x: 1 if isinstance(x, dict) and x.get('city') else 0)
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['text_clean'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine all features
        numeric_features = df[['text_length', 'word_count', 'sentence_count', 
                               'title_length', 'title_word_count', 
                               'source_encoded', 'has_company', 'has_location']]
        
        features_df = pd.concat([numeric_features, tfidf_df], axis=1)
        self.feature_names = list(features_df.columns)
        
        return features_df
    
    def load_data(self, file_path, max_samples=None):
        """Load and preprocess data from JSONL file"""
        print(f"Loading data from {file_path}...")
        items = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    # Only include items with department information
                    if 'position' in item and isinstance(item['position'], dict):
                        if 'department' in item['position'] and item['position']['department']:
                            items.append(item)
                except:
                    continue
        
        print(f"Loaded {len(items)} job postings with department information")
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Extract target variable
        df['department'] = df['position'].apply(
            lambda x: x.get('department') if isinstance(x, dict) else None
        )
        
        # Remove rows with missing target
        df = df[df['department'].notna()]
        
        # Filter out classes with too few samples (optional)
        dept_counts = df['department'].value_counts()
        min_samples = 10  # Minimum samples per class
        valid_departments = dept_counts[dept_counts >= min_samples].index
        df = df[df['department'].isin(valid_departments)]
        
        print(f"Final dataset size: {len(df)}")
        print(f"Department distribution:\n{df['department'].value_counts()}")
        
        return df
    
    def train(self, file_path, max_samples=10000):
        """Train the model"""
        # Load data
        df = self.load_data(file_path, max_samples)
        
        # Extract features
        print("\nExtracting features...")
        X = self.extract_features(df)
        y = df['department'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=42, stratify=y_encoded
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Calculate class weights for imbalance handling
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # Initialize and train XGBoost model
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Detailed classification report
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            y_test, y_test_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Confusion matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        print(cm_df)
        
        # Feature importance
        print("\n" + "="*80)
        print("TOP 20 MOST IMPORTANT FEATURES")
        print("="*80)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(20).to_string(index=False))
        
        return {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'model': self.model,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
    
    def predict(self, text, name="", source=""):
        """Predict department for a new job posting"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create a single-row DataFrame
        data = {
            'text': [text],
            'name': [name],
            'source': [source],
            'orgCompany': [{}],
            'orgAddress': [{}]
        }
        df = pd.DataFrame(data)
        
        # Extract features
        X = self.extract_features(df)
        
        # Predict
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Get probabilities
        y_proba = self.model.predict_proba(X)[0]
        proba_dict = {
            self.label_encoder.classes_[i]: proba
            for i, proba in enumerate(y_proba)
        }
        
        return y_pred[0], proba_dict


if __name__ == "__main__":
    # Initialize classifier
    classifier = JobDepartmentClassifier(max_features=5000)
    
    # Train model
    results = classifier.train(
        file_path="techmap-jobs_us_2023-05-05.json",
        max_samples=10000  # Adjust based on your dataset size
    )
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    
    if results['test_accuracy'] >= 0.90:
        print("\n✓ Target accuracy of 90% achieved!")
    else:
        print("\n⚠ Target accuracy not yet achieved. Consider:")
        print("  - Increasing max_samples for more training data")
        print("  - Tuning hyperparameters further")
        print("  - Trying BERT/RoBERTa for higher accuracy")
    
    # Example prediction
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    sample_text = """
    We are looking for a Senior Software Engineer to join our development team.
    You will be responsible for designing and implementing scalable web applications
    using Python, Django, and React. Experience with cloud platforms (AWS) is required.
    """
    pred, proba = classifier.predict(sample_text, name="Senior Software Engineer")
    print(f"Predicted Department: {pred}")
    print(f"Confidence: {proba[pred]:.4f}")
    print(f"\nAll probabilities:")
    for dept, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dept}: {prob:.4f}")

