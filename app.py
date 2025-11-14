"""
Flask Web Application for Job Department Classifier
Run this file to test the trained XGBoost model through a web interface
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import re

app = Flask(__name__)

# Load the trained model and preprocessing objects
print("Loading model and preprocessing objects...")
with open('xgboost_department_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model loaded successfully!")

def clean_text(text):
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

def extract_features(text, title="", source=""):
    """Extract features from job posting"""
    # Create DataFrame
    df = pd.DataFrame({
        'text': [text],
        'name': [title],
        'source': [source],
        'orgCompany': [{}],
        'orgAddress': [{}]
    })
    
    # Text features
    df['text_clean'] = df['text'].apply(clean_text)
    df['text_length'] = df['text_clean'].str.len()
    df['word_count'] = df['text_clean'].str.split().str.len()
    df['sentence_count'] = df['text_clean'].str.count(r'[.!?]+')
    
    # Title features
    df['title_clean'] = df['name'].apply(clean_text)
    df['title_length'] = df['title_clean'].str.len()
    df['title_word_count'] = df['title_clean'].str.split().str.len()
    
    # Source encoding
    df['source_encoded'] = pd.Categorical(df['source']).codes
    
    # Company and location features
    df['has_company'] = df['orgCompany'].apply(lambda x: 1 if isinstance(x, dict) and x.get('name') else 0)
    df['has_location'] = df['orgAddress'].apply(lambda x: 1 if isinstance(x, dict) and x.get('city') else 0)
    
    # TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(df['text_clean'])
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])],
        index=df.index
    )
    
    # Combine all features
    numeric_features = df[['text_length', 'word_count', 'sentence_count', 
                           'title_length', 'title_word_count', 
                           'source_encoded', 'has_company', 'has_location']].reset_index(drop=True)
    
    features_df = pd.concat([numeric_features.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    return features_df

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data
        job_description = request.form.get('job_description', '')
        job_title = request.form.get('job_title', '')
        source = request.form.get('source', 'web')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Extract features
        X = extract_features(job_description, job_title, source)
        
        # Make prediction
        y_pred_encoded = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Get probabilities
        y_proba = model.predict_proba(X)[0]
        proba_dict = {
            label_encoder.classes_[i]: float(proba)
            for i, proba in enumerate(y_proba)
        }
        
        # Sort probabilities
        sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'success': True,
            'predicted_department': y_pred[0],
            'confidence': float(proba_dict[y_pred[0]]),
            'all_probabilities': sorted_probs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("JOB DEPARTMENT CLASSIFIER WEB APP")
    print("="*80)
    print("Server starting...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

