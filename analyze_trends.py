"""
Analyze job posting trends from the dataset
Prepare data for time series forecasting and career path analysis
"""

import json
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading job postings data...")
data = []
with open('techmap-jobs_us_2023-05-05.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            job = json.loads(line.strip())
            data.append(job)
        except:
            continue

print(f"Loaded {len(data)} job postings")

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract key information
print("\n" + "="*80)
print("DATA ANALYSIS FOR PREDICTIVE MODELS")
print("="*80)

# 1. Department Distribution
print("\n1. DEPARTMENT DISTRIBUTION")
if 'position' in df.columns:
    df['department'] = df['position'].apply(
        lambda x: x.get('department') if isinstance(x, dict) else None
    )
    dept_counts = df['department'].value_counts()
    print(dept_counts)

# 2. Date Analysis
print("\n2. DATE ANALYSIS")
if 'datePosted' in df.columns:
    df['datePosted'] = pd.to_datetime(df['datePosted'], errors='coerce')
    print(f"Date range: {df['datePosted'].min()} to {df['datePosted'].max()}")
    print(f"Valid dates: {df['datePosted'].notna().sum()}")
    
# 3. Job Title Analysis (for career paths)
print("\n3. JOB TITLES ANALYSIS")
if 'name' in df.columns:
    # Common titles
    titles = df['name'].value_counts().head(20)
    print("Top 20 job titles:")
    print(titles)

# 4. Seniority Levels (extract from titles)
print("\n4. SENIORITY LEVELS")
seniority_keywords = {
    'junior': ['junior', 'jr', 'entry', 'associate'],
    'mid': ['mid', 'intermediate'],
    'senior': ['senior', 'sr', 'lead', 'principal'],
    'manager': ['manager', 'director', 'vp', 'head', 'chief']
}

def extract_seniority(title):
    if not isinstance(title, str):
        return 'unknown'
    title_lower = title.lower()
    for level, keywords in seniority_keywords.items():
        if any(kw in title_lower for kw in keywords):
            return level
    return 'mid'  # default

df['seniority'] = df['name'].apply(extract_seniority)
print(df['seniority'].value_counts())

# 5. Skills Analysis
print("\n5. SKILLS EXTRACTION")
common_skills = [
    'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
    'kubernetes', 'machine learning', 'ai', 'data science', 'salesforce',
    'excel', 'powerpoint', 'communication', 'leadership', 'management'
]

def extract_skills(text):
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return found_skills

df['skills'] = df['text'].apply(extract_skills)
all_skills = [skill for skills_list in df['skills'] for skill in skills_list]
skill_counts = Counter(all_skills)
print("Top 15 skills mentioned:")
for skill, count in skill_counts.most_common(15):
    print(f"  {skill}: {count}")

# 6. Create time series data for forecasting
print("\n6. TIME SERIES DATA PREPARATION")
if 'datePosted' in df.columns and df['datePosted'].notna().sum() > 0:
    df_with_dates = df[df['datePosted'].notna()].copy()
    df_with_dates['date'] = df_with_dates['datePosted'].dt.date
    
    # Jobs per department over time
    dept_time_series = df_with_dates.groupby(['date', 'department']).size().reset_index(name='count')
    dept_time_series.to_csv('department_time_series.csv', index=False)
    print(f"Created department_time_series.csv with {len(dept_time_series)} records")
    
    # Overall jobs over time
    overall_time_series = df_with_dates.groupby('date').size().reset_index(name='count')
    overall_time_series.to_csv('overall_time_series.csv', index=False)
    print(f"Created overall_time_series.csv with {len(overall_time_series)} records")

# 7. Career Path Data
print("\n7. CAREER PATH DATA")
career_paths = []
for dept in df['department'].unique():
    if pd.notna(dept):
        dept_df = df[df['department'] == dept]
        for seniority in ['junior', 'mid', 'senior', 'manager']:
            count = (dept_df['seniority'] == seniority).sum()
            if count > 0:
                career_paths.append({
                    'department': dept,
                    'current_level': seniority,
                    'count': count
                })

career_df = pd.DataFrame(career_paths)
career_df.to_csv('career_paths_data.csv', index=False)
print(f"Created career_paths_data.csv with {len(career_df)} records")

# 8. Skills by Department
print("\n8. SKILLS BY DEPARTMENT")
dept_skills = defaultdict(Counter)
for idx, row in df.iterrows():
    if pd.notna(row.get('department')) and row.get('skills'):
        dept_skills[row['department']].update(row['skills'])

skills_by_dept = []
for dept, skills in dept_skills.items():
    for skill, count in skills.most_common(10):
        skills_by_dept.append({
            'department': dept,
            'skill': skill,
            'count': count
        })

skills_df = pd.DataFrame(skills_by_dept)
skills_df.to_csv('department_skills.csv', index=False)
print(f"Created department_skills.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. department_time_series.csv - For trend forecasting")
print("  2. overall_time_series.csv - Overall job market trends")
print("  3. career_paths_data.csv - For career progression modeling")
print("  4. department_skills.csv - Skills required by department")
print("\nNext step: Building prediction models...")

