<!-- c798fa05-6c93-4811-81be-6c631056fa97 f1ad85c4-0c3e-4ce7-be3a-54328d75880f -->
# Train XGBoost Job Department Classifier

## Configuration

- **Dataset**: techmap-jobs_us_2023-05-05.json
- **Samples**: 50,000 job postings
- **Target**: Department classification (7 classes)
- **Model**: XGBoost with TF-IDF features
- **Goal**: 90%+ test accuracy

## Implementation Steps

### 1. Update Training Script

Enhance `train_model.py` with:

- Increase `max_samples=50000` for better accuracy
- Add progress tracking and detailed logging
- Implement model/vectorizer saving functionality
- Add cross-validation (5-fold stratified)
- Include confidence thresholds for predictions

### 2. Data Preprocessing

- Load 50,000 samples from JSONL file
- Filter for valid department labels
- Clean text (remove HTML, special chars, normalize)
- Handle class imbalance with balanced weights
- Split: 70% train, 15% validation, 15% test

### 3. Feature Engineering

- TF-IDF vectorization (5000 features, 1-2 grams)
- Extract metadata: text length, word count, sentence count
- Title features: length, word count
- Encode categorical: source, company presence, location
- Combine into feature matrix (~5008 features)

### 4. Model Training

Use pre-optimized XGBoost hyperparameters:

- n_estimators=300
- max_depth=6
- learning_rate=0.1
- subsample=0.8
- colsample_bytree=0.8
- Early stopping on validation set
- Balanced sample weights for class imbalance

### 5. Evaluation & Results

Generate comprehensive metrics:

- Accuracy (primary metric)
- Precision, Recall, F1-score per class
- Confusion matrix
- Feature importance (top 20)
- Cross-validation scores

### 6. Model Persistence

Save trained artifacts:

- XGBoost model (pickle/joblib)
- TF-IDF vectorizer
- Label encoder
- Feature names and metadata

## Expected Outcome

- Test accuracy: 90-95%
- Training time: 15-30 minutes
- Saved model ready for deployment

### To-dos

- [ ] Update train_model.py with 50K samples, progress tracking, and model saving
- [ ] Execute training script and monitor progress
- [ ] Review accuracy metrics and determine if 90% threshold is met
- [ ] Ensure model, vectorizer, and encoder are properly saved
- [ ] 