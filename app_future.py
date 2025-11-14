"""
Flask Web App for Future Job Predictions
- Job Market Trend Forecasting
- Career Path Prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load models
print("Loading future prediction models...")
forecast_df = pd.read_csv('job_market_forecast.csv')
career_df = pd.read_csv('career_progressions.csv')

with open('career_path_model.pkl', 'rb') as f:
    career_paths = pickle.load(f)

print("Models loaded successfully!")

@app.route('/future')
def future_home():
    """Render the future predictions page"""
    return render_template('future.html')

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """Get job market forecast"""
    try:
        data = request.json
        department = data.get('department', 'ALL').upper()
        months = int(data.get('months', 12))
        
        if department == 'ALL':
            result_df = forecast_df
        else:
            result_df = forecast_df[forecast_df['department'] == department]
        
        # Get unique dates and limit
        unique_dates = sorted(result_df['date'].unique())[:months]
        
        # Prepare forecast data
        forecast_data = []
        for date in unique_dates:
            date_data = result_df[result_df['date'] == date]
            for _, row in date_data.iterrows():
                forecast_data.append({
                    'date': row['date'],
                    'department': row['department'],
                    'predicted_jobs': int(row['predicted_jobs']),
                    'growth_rate': float(row['growth_rate'])
                })
        
        return jsonify({'success': True, 'forecast': forecast_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/career-path', methods=['POST'])
def get_career_path():
    """Get career path prediction"""
    try:
        data = request.json
        department = data.get('department', '').upper()
        current_level = data.get('level', '').lower()
        
        if department not in career_paths:
            return jsonify({
                'success': False,
                'error': f'Department not found. Available: {", ".join(career_paths.keys())}'
            }), 400
        
        if current_level not in career_paths[department]:
            return jsonify({
                'success': False,
                'error': f'Level not found. Available: {", ".join(career_paths[department].keys())}'
            }), 400
        
        progression = career_paths[department][current_level]
        
        # Build full career path
        path = []
        current = current_level
        while current in career_paths[department]:
            prog = career_paths[department][current]
            path.append({
                'level': current,
                'next_level': prog['next'],
                'probability': float(prog['probability']),
                'years': float(prog['years'])
            })
            current = prog['next']
            if current == 'manager':
                break
        
        return jsonify({
            'success': True,
            'current_level': current_level,
            'department': department,
            'next_level': progression['next'],
            'probability': float(progression['probability']),
            'typical_years': float(progression['years']),
            'full_path': path
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/growth-rankings', methods=['GET'])
def get_growth_rankings():
    """Get department growth rankings"""
    try:
        growth_data = forecast_df[['department', 'growth_rate']].drop_duplicates()
        growth_data = growth_data.sort_values('growth_rate', ascending=False)
        
        rankings = []
        for _, row in growth_data.iterrows():
            rankings.append({
                'department': row['department'],
                'growth_rate': float(row['growth_rate'])
            })
        
        return jsonify({'success': True, 'rankings': rankings})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FUTURE JOB PREDICTION WEB APP")
    print("="*80)
    print("Server starting...")
    print("Open your browser and navigate to: http://localhost:5001/future")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)

