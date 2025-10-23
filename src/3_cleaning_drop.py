import pandas as pd
import os

df_raw = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")

print("\nLoaded raw shape:", df_raw.shape)

df_clean = df_raw.copy()  
df_clean = df_clean.drop(columns=['id', 'Residence_type', 'gender']) # removing columns
print("After drop cols, shape:", df_clean.shape)

# binary variable ever_married (also hypertension, heart_disease, stroke) encoded as 0/1 for ML compatibility.
if 'ever_married' in df_clean.columns: # replace 'Yes'/'No' in subcategories with 1/0
    df_clean['ever_married'] = df_clean['ever_married'].map({'Yes': 1, 'No': 0})
    print("Converted 'ever_married' → 1 for Yes, 0 for No")

os.makedirs("data/processed", exist_ok=True)    
df_clean.to_csv("data/processed/df_clean.csv", index=False)  
print("Saved cleaned dataset as data/processed/df_clean.csv")
