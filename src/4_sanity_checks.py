import pandas as pd
import numpy as np

df_clean = pd.read_csv("data/processed/df_clean.csv")   

print("\nLoaded df_clean:", df_clean.shape)

print("\nNumeric columns:", df_clean.select_dtypes(include=np.number).columns.tolist())  # verify if numeric cols - no text values
for col in df_clean.select_dtypes(exclude='object').columns:
    non_numeric = df_clean[~df_clean[col].apply(lambda x: pd.api.types.is_number(x))].shape[0]
    print(f"{col}: {non_numeric} non-numeric values")

print("\nText columns:", df_clean.select_dtypes(include='object').columns.tolist())  # verify if text cols - no numeric values
for col in df_clean.select_dtypes(include='object').columns:
    numeric_values = df_clean[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).sum()
    print(f"{col}: {numeric_values} numeric values")

dup_count = df_clean.duplicated().sum() # duplicates
print(f'\nDuplicates:{dup_count}')

AGE_MIN, AGE_MAX = 0, 100  # sanity check - realistic values ranges, glucose: 30-800 | age: 0-100 | BMI: 10-100
BMI_MIN, BMI_MAX = 10, 100
GLUCOSE_MIN, GLUCOSE_MAX = 30, 800

out_age = df_clean[(df_clean['age'] < AGE_MIN) | (df_clean['age'] > AGE_MAX)]
out_bmi = df_clean[(df_clean['bmi'] < BMI_MIN) | (df_clean['bmi'] > BMI_MAX)]
out_glucose = df_clean[(df_clean['avg_glucose_level'] < GLUCOSE_MIN) | (df_clean['avg_glucose_level'] > GLUCOSE_MAX)]

print(f"\nValues out of range - age: {len(out_age)} records")
print(f"Values out of range - BMI: {len(out_bmi)} records")
print(f"Values out of range - glucose level: {len(out_glucose)} records")

# check logical inconsistencies
inconsistent_work = df_clean[(df_clean['age'] < 16) & (~df_clean['work_type'].isin(['children', 'Never_worked']))]
inconsistent_married = df_clean[(df_clean['age'] < 18) & (df_clean['ever_married'] == 1)]
inconsistent_bmi = df_clean[(df_clean['age'] < 5) & (df_clean['bmi'] > 30)]
inconsistent_smoking = df_clean[(df_clean['work_type'] == 'children') & (df_clean['smoking_status'].isin(['smokes', 'formerly smoked']))]
inconsistent_stroke = df_clean[(df_clean['stroke'] == 1) & (df_clean['age'] < 10)]

print("\nChildren with non-child work type:", inconsistent_work.shape[0])
print("Children marked as married:", inconsistent_married.shape[0])
print("Children with unrealistic BMI:", inconsistent_bmi.shape[0])
print("Children with smoking history:", inconsistent_smoking.shape[0])
print("Children with stroke:", inconsistent_stroke.shape[0])

print("\nAge distribution of children with invalid work_type:") # data inconsistency, potential anomaly
print(inconsistent_work['age'].describe())
print("\nWork_type distribution among children under 16:")
print(inconsistent_work['work_type'].value_counts())

print("\nAge distribution of children with smoking history:")
print(inconsistent_smoking['age'].describe())
print("\nsmoking_status distribution among children under 16:")
print(inconsistent_smoking['smoking_status'].value_counts())

# Found 60 children (<16) with non-child work_type assignments and 15 with smoking history (ages 10–15).
# These are likely data entry inconsistencies but not necessarily invalid.

# check medical and further logical inconsistencies
young_hyper = df_clean[(df_clean['age'] < 18) & (df_clean['hypertension'] == 1)]  # small irregularities observed but logical
young_heart = df_clean[(df_clean['age'] < 18) & (df_clean['heart_disease'] == 1)]
print("\nChildren (<18) with hypertension:", young_hyper.shape[0])
print("Children (<18) with heart disease:", young_heart.shape[0])

high_bmi_kids = df_clean[(df_clean['age'] < 10) & (df_clean['bmi'] > 40)]  # result is logically consistent
print("\nChildren (<10) with BMI > 40:", high_bmi_kids.shape[0])

print("\nGlucose distribution in patients with/out stroke-if reversed relationships")  # result is logically consistent
print(df_clean.groupby('stroke')['avg_glucose_level'].describe())

df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 18, 40, 60, 100],
    labels=['Child', 'Young Adult', 'Middle-aged', 'Senior']
)
print("\nHypertension rate by age group and work type (%):")  # result is logically consistent
print(df_clean.groupby(['age_group', 'work_type'], observed=True)['hypertension'].mean().unstack() * 100)