import pandas as pd
from sklearn.model_selection import train_test_split
import os

df_clean = pd.read_csv("data/processed/df_clean.csv") 

print(f"\nLoaded cleaned dataset: {df_clean.shape}")

df_train, df_test = train_test_split(   # splitting train/ test
    df_clean,
    test_size=0.2,               # 20% for testing
    random_state=42,             # reproducibility
    stratify=df_clean["stroke"]  # maintain class balance, specially important here, since stroke cases (1) are much less frequent
)

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

print("\nStroke distribution (%):")   # stroke distribution comparison
print("Train:", df_train["stroke"].mean() * 100)
print("Test :", df_test["stroke"].mean() * 100)

if set(df_train.columns) == set(df_test.columns):   # confirm column consistency
    print("Column structure consistent between train and test.")
else:
    print("Column mismatch detected — check data cleaning steps.")

df_train.to_csv("data/processed/df_train.csv", index=False)    
df_test.to_csv("data/processed/df_test.csv", index=False)
print("\nTrain/test data saved in 'data/processed/' folder")
