"""
Future Job Prediction CLI Tool
Use the trained models to make predictions
"""

import pickle
import pandas as pd
import sys

# Load models
print("Loading models...")
forecast_df = pd.read_csv('job_market_forecast.csv')
career_df = pd.read_csv('career_progressions.csv')

with open('career_path_model.pkl', 'rb') as f:
    career_paths = pickle.load(f)

print("Models loaded successfully!\n")

def predict_job_market(department=None, months_ahead=12):
    """Predict job market trends"""
    print("="*80)
    print("JOB MARKET FORECAST")
    print("="*80)
    
    if department:
        data = forecast_df[forecast_df['department'] == department.upper()]
        if len(data) == 0:
            print(f"Department '{department}' not found.")
            return
    else:
        data = forecast_df
    
    # Get forecast for specified months ahead
    unique_dates = sorted(data['date'].unique())
    forecast_dates = unique_dates[:months_ahead]
    
    for date in forecast_dates:
        date_data = data[data['date'] == date]
        print(f"\n{date}:")
        for _, row in date_data.iterrows():
            print(f"  {row['department']}: {row['predicted_jobs']} jobs (Growth: +{row['growth_rate']*100:.0f}%/year)")

def predict_career_path(department, current_level):
    """Predict career progression"""
    print("="*80)
    print("CAREER PATH PREDICTION")
    print("="*80)
    
    dept = department.upper()
    level = current_level.lower()
    
    if dept not in career_paths:
        print(f"Department '{department}' not found.")
        print(f"Available: {', '.join(career_paths.keys())}")
        return
    
    if level not in career_paths[dept]:
        print(f"Level '{current_level}' not found for {dept}")
        print(f"Available: {', '.join(career_paths[dept].keys())}")
        return
    
    progression = career_paths[dept][level]
    
    print(f"\nCurrent Position: {level.upper()} in {dept}")
    print(f"Next Level: {progression['next'].upper()}")
    print(f"Probability: {progression['probability']*100:.0f}%")
    print(f"Typical Timeline: {progression['years']} years")
    
    # Show full career path
    print(f"\n{dept} Career Path:")
    current = level
    path = [current]
    while current in career_paths[dept]:
        next_level = career_paths[dept][current]['next']
        path.append(next_level)
        current = next_level
        if current == 'manager':
            break
    
    print(" -> ".join([p.upper() for p in path]))

def show_growth_rankings():
    """Show departments ranked by growth"""
    print("="*80)
    print("DEPARTMENT GROWTH RANKINGS (Next 2 Years)")
    print("="*80)
    
    growth_rates = forecast_df[['department', 'growth_rate']].drop_duplicates()
    growth_rates = growth_rates.sort_values('growth_rate', ascending=False)
    
    for i, (_, row) in enumerate(growth_rates.iterrows(), 1):
        print(f"{i}. {row['department']}: +{row['growth_rate']*100:.0f}% annually")

def interactive_mode():
    """Interactive prediction mode"""
    while True:
        print("\n" + "="*80)
        print("FUTURE JOB PREDICTION SYSTEM")
        print("="*80)
        print("\n1. Job Market Forecast (by department)")
        print("2. Career Path Prediction")
        print("3. Growth Rankings")
        print("4. Full Market Overview")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ").strip()
        
        if choice == '1':
            dept = input("Enter department (or press Enter for all): ").strip()
            months = input("Months ahead (default 12): ").strip()
            months = int(months) if months else 12
            predict_job_market(dept if dept else None, months)
            
        elif choice == '2':
            dept = input("Enter department (e.g., DEVELOPMENT): ").strip()
            level = input("Enter current level (junior/mid/senior/manager): ").strip()
            predict_career_path(dept, level)
            
        elif choice == '3':
            show_growth_rankings()
            
        elif choice == '4':
            predict_job_market(None, 6)
            
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]
        
        if command == "forecast":
            dept = sys.argv[2] if len(sys.argv) > 2 else None
            predict_job_market(dept)
            
        elif command == "career":
            if len(sys.argv) < 4:
                print("Usage: python predict_future.py career DEPARTMENT LEVEL")
                print("Example: python predict_future.py career DEVELOPMENT junior")
            else:
                predict_career_path(sys.argv[2], sys.argv[3])
                
        elif command == "growth":
            show_growth_rankings()
    else:
        # Interactive mode
        interactive_mode()

