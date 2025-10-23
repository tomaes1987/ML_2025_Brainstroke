import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df_raw = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")  

print("Data preview:") # quick stats
print(df_raw.head())

print("\nData Information:")
print(df_raw.info())

# binary categorical variables (e.g., hypertension, heart_disease and target stroke) 
# were encoded as integers 0 and 1 to facilitate numerical processing and modeling
# the choice of integer representation maintains interpretability (0 = No, 1 = Yes) 
# while ensuring compatibility with ML algorithms

# --- komentarz do kodu poniżej ---
# dla int64 pandas pokazuje statystyki liczbowe, co nie ma sensu dla kolumn binarnych (0/1), dlateg
# konwertujemy je na kategorieczne
#------------ kod do implementacji ------------
"""categorical_columns = ['hypertension', 'heart_disease', 'stroke']
for col in categorical_columns:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].astype('category')"""
#------------ koniec kodu do usunięcia ------------
print("\nNumerical statistics:")
print(df_raw.describe())

# --- komentarz do kodu powyżej ---
# dla int64 pandas pokazuje statystyki liczbowe, co nie ma sensu dla kolumn binarnych (0/1)

# quick stats to determine column relevance (gender)
print("\nAverage values ​​for gender (comparison):")
gender_stats = df_raw.groupby('gender')[['avg_glucose_level', 'bmi', 'stroke', 'age']].agg(['mean', 'std', 'count'])
print(gender_stats)

# confirmation that there is no significant difference in the incidence of stroke 
# between the gender and Residence_type column categories
gender_stroke_rate = df_raw.groupby('gender')['stroke'].mean() * 100
print("\n Proportion of stroke (%) for each gender:")
print(gender_stroke_rate)
residence_stroke_rate = df_raw.groupby('Residence_type')['stroke'].mean() * 100
print("\n Proportion of stroke (%) for each Residence_type:")
print(residence_stroke_rate)

# Chi-square test for gender vs stroke
gender_ct = pd.crosstab(df_raw['gender'], df_raw['stroke'])
chi2_gender, p_gender, dof_gender, ex_gender = chi2_contingency(gender_ct)
print("\nGender vs stroke → p-value:", p_gender)

# Chi-square test for Residence_type vs stroke
res_ct = pd.crosstab(df_raw['Residence_type'], df_raw['stroke'])
chi2_res, p_res, dof_res, ex_res = chi2_contingency(res_ct)
print("Residence_type vs stroke → p-value:", p_res)

# confirmation that the gender column is very weakly correlated with other features including stroke
df_temp = df_raw.copy()  
df_gender_num = df_temp[df_temp['gender'].isin(['Male', 'Female'])].copy()
df_gender_num['gender_num'] = np.where(df_gender_num['gender'] == 'Male', 1, 0) # binary encoding
corr_glucose = df_gender_num[['gender_num', 'avg_glucose_level']].corr().iloc[0,1]
corr_bmi = df_gender_num[['gender_num', 'bmi']].corr().iloc[0,1]
corr_stroke = df_gender_num[['gender_num', 'stroke']].corr().iloc[0,1]
corr_age = df_gender_num[['gender_num', 'age']].corr().iloc[0,1]

print("\nGender correlations with other variables:")
print(f"gender ↔ avg_glucose_level: {corr_glucose:.3f}")
print(f"gender ↔ bmi: {corr_bmi:.3f}")
print(f"gender ↔ stroke: {corr_stroke:.3f}")
print(f"gender ↔ age: {corr_age:.3f}") 

# correlation for Residence_type
df_temp['Residence_type_num'] = df_temp['Residence_type'].map({'Urban': 1, 'Rural': 0})
numeric_cols = df_temp.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop('id', errors='ignore')
corrs = df_temp[numeric_cols].corr()['Residence_type_num'].sort_values(ascending=False)
print("\n Correlations of Residence_type (Urban=1, Rural=0) with numeric features:")
print(corrs)

plt.figure(figsize=(8,5))
sns.barplot(x=corrs.values, y=corrs.index)
plt.title("Correlation of Residence_type (Urban=1, Rural=0) with other numeric features")
plt.xlabel("Correlation value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# decision based on the above and sweetviz - columns to be removed: 'id', 'Residence_type' and 'gender'