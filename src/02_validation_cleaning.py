import numpy as np
import pandas as pd
import os

df = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")
df_clean = df.copy()  

# verify if numeric cols - no text values
print("\nNumeric columns:", df_clean.select_dtypes(include=np.number).columns.tolist())  
for col in df_clean.select_dtypes(exclude='object').columns:
    non_numeric = df_clean[~df_clean[col].apply(lambda x: pd.api.types.is_number(x))].shape[0]
    print(f"{col}: {non_numeric} non-numeric values")

# verify if text cols - no numeric values
print("\nText columns:", df_clean.select_dtypes(include='object').columns.tolist())  
for col in df_clean.select_dtypes(include='object').columns:
    numeric_values = df_clean[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).sum()
    print(f"{col}: {numeric_values} numeric values")

# check if duplicates
if df_clean.drop(columns=['id']).duplicated().any():
    print("Warning: Duplicates found in the dataset without 'id' column.")
    print("Number of duplicate rows (without 'id'):", df_clean.drop(columns=['id']).duplicated().sum())
elif df_clean.duplicated().any():
    print("Warning: Duplicates found in the dataset.")
    print("Number of duplicate rows:", df_clean.duplicated().sum())
else: 
    print("No duplicates found in the dataset")

dup_count = df_clean.duplicated().sum() 
print(f'\nDuplicates:{dup_count}')

# remove 'id' column
df_clean = df_clean.drop(columns=['id'])
# check if dropped
if 'id' in df_clean.columns:
    print("Error: 'id' column was not dropped.")
else:
    print("'id' column successfully dropped.")

# sanity check - realistic values ranges, glucose: 30-800 | age: 0-100 | BMI: 10-100
AGE_MIN, AGE_MAX = 0, 100  
BMI_MIN, BMI_MAX = 10, 100
GLUCOSE_MIN, GLUCOSE_MAX = 30, 800

out_age = df_clean[(df_clean['age'] < AGE_MIN) | (df_clean['age'] > AGE_MAX)]
out_bmi = df_clean[(df_clean['bmi'] < BMI_MIN) | (df_clean['bmi'] > BMI_MAX)]
out_glucose = df_clean[(df_clean['avg_glucose_level'] < GLUCOSE_MIN) | (df_clean['avg_glucose_level'] > GLUCOSE_MAX)]

print(f"\nValues out of range - age: {len(out_age)} records")
print(f"Values out of range - BMI: {len(out_bmi)} records")
print(f"Values out of range - glucose level: {len(out_glucose)} records")

# check logical inconsistencies
# children with non-child work types
inconsistent_work = df_clean[(df_clean['age'] < 16) & (~df_clean['work_type'].isin(['children', 'Never_worked']))]
# married children
inconsistent_married = df_clean[(df_clean['age'] < 18) & (df_clean['ever_married'] == 1)]
# unrealistic BMI
inconsistent_bmi = df_clean[(df_clean['age'] < 5) & (df_clean['bmi'] > 30)]
# smoking history
inconsistent_smoking = df_clean[(df_clean['work_type'] == 'children') & (df_clean['smoking_status'].isin(['smokes', 'formerly smoked']))]
# stroke occurrences
inconsistent_stroke = df_clean[(df_clean['stroke'] == 1) & (df_clean['age'] < 10)]

print("\nChildren with non-child work type:", inconsistent_work.shape[0])
print("Children marked as married:", inconsistent_married.shape[0])
print("Children with unrealistic BMI:", inconsistent_bmi.shape[0])
print("Children with smoking history:", inconsistent_smoking.shape[0])
print("Children with stroke:", inconsistent_stroke.shape[0])


print("\nAge distribution of children with non-child work type:") 
print(inconsistent_work['age'].describe())

print("\nWork type distribution among children under 16:")
print(inconsistent_work['work_type'].value_counts())

print("\nAge distribution of children with smoking history:")
print(inconsistent_smoking['age'].describe())
print("\nsmoking_status distribution among children under 16:")
print(inconsistent_smoking['smoking_status'].value_counts())

# Found 60 children (<16) with non-child work type assignments and 15 with smoking history (ages 10–15).
# These are likely data entry inconsistencies but not necessarily invalid.

# show only if strokes class 1 found in inconsistent_work,
# which would be more concerning, because stroke is a class that is imbalanced
inconsistent_work_stroke = inconsistent_work[inconsistent_work['stroke'] == 1]
if not inconsistent_work_stroke.empty:
    print("\nChildren with invalid work type who had strokes:")
    print(inconsistent_work_stroke)

# check medical and further logical inconsistencies


young_hyper = df_clean[(df_clean['age'] < 18) & (df_clean['hypertension'] == 1)]  
young_heart = df_clean[(df_clean['age'] < 18) & (df_clean['heart_disease'] == 1)]
print("\nChildren (<18) with hypertension:", young_hyper.shape[0])
print("Children (<18) with heart disease:", young_heart.shape[0])
# small irregularities observed but logically consistent


high_bmi_kids = df_clean[(df_clean['age'] < 10) & (df_clean['bmi'] > 40)]  
print("\nChildren (<10) with BMI > 40:", high_bmi_kids.shape[0])
# result is logically consistent


print("\nGlucose distribution in patients with and without stroke, if reverersed logic exists, would be concerning:")  
print(df_clean.groupby('stroke')['avg_glucose_level'].describe())
# result is logically consistent


df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 18, 40, 60, 100],
    labels=['Child', 'Young Adult', 'Middle-aged', 'Senior']
    )
print("\nHypertension rate by age group and work type (%):")  
print(df_clean.groupby(['age_group', 'work_type'], observed=True)['hypertension'].mean().unstack() * 100)
# result is logically consistent

df_clean = df_clean.drop(columns=['age_group'])

print(df_clean.info())

os.makedirs("data/processed", exist_ok=True)    
df_clean.to_csv("data/processed/df_clean.csv", index=False)