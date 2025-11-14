# 🌐 Job Department Classifier - Web Interface

## Quick Start

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open Your Browser

Navigate to: **http://localhost:5000**

### 3. Test the Model

The web interface provides:
- ✨ Beautiful, modern UI
- 📝 Input form for job descriptions
- 🎯 Real-time predictions
- 📊 Probability visualization for all departments
- 🔄 Pre-loaded example jobs (Developer, Sales, Finance, HR)

## Features

### Input Fields
- **Job Title** (optional): The title of the position
- **Job Description** (required): Full job posting text

### Output Display
- **Predicted Department**: The most likely department
- **Confidence Score**: How confident the model is (0-100%)
- **Probability Bar Chart**: Visual breakdown of all department probabilities

### Example Jobs
Click the example buttons to instantly load sample job postings:
- 💻 **Developer** - Software engineering role
- 💰 **Sales** - Account executive position
- 📊 **Finance** - Financial analyst role
- 👥 **HR** - Human resources manager

## How It Works

1. The Flask server loads the trained model files:
   - `xgboost_department_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`

2. When you submit a job description:
   - Text is cleaned and preprocessed
   - Features are extracted (TF-IDF + metadata)
   - XGBoost model makes prediction
   - Results are displayed with confidence scores

3. The interface shows:
   - Top prediction with confidence
   - All 7 department probabilities with visual bars
   - Smooth animations and modern design

## Technical Details

### Backend (app.py)
- **Framework**: Flask
- **Model**: XGBoost Classifier
- **Features**: 5,008 dimensions (TF-IDF + metadata)
- **API Endpoint**: `/predict` (POST)

### Frontend (templates/index.html)
- **Design**: Modern gradient UI with animations
- **Technology**: Pure HTML/CSS/JavaScript (no frameworks)
- **Responsive**: Works on desktop and mobile
- **Real-time**: AJAX-based predictions

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Files

- `app.py` - Flask backend application
- `templates/index.html` - Web interface
- Model files (automatically loaded):
  - `xgboost_department_model.pkl`
  - `tfidf_vectorizer.pkl`
  - `label_encoder.pkl`

## Model Performance

The web app uses a model trained on 4,612 job postings with:
- ✅ **95.23% Test Accuracy**
- ✅ **7 Department Classes**
- ✅ **5,008 Features**

## Departments

The model can classify jobs into:
1. DEVELOPMENT
2. FINANCE
3. HR
4. MARKETING
5. RESEARCH
6. SALES
7. SERVICE

---

Enjoy testing your AI-powered job classifier! 🎉

