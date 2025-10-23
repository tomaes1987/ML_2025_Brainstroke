import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import joblib
import os

df_train = pd.read_csv("data/processed/df_train.csv") 

print(f"\nLoaded train set")

print("\nMissing values in the train set before imputing::")
print(df_train.isnull().sum())

df_knn = df_train.copy()
excluded_cols = ['stroke', 'bmi']
categorical_cols = ['work_type', 'smoking_status']             # encoding (dummy variables)
df_knn = pd.get_dummies(df_knn, columns=categorical_cols, drop_first=True)    

# KNN Imputation
features_for_knn = df_knn.columns.drop(excluded_cols)

imputer = KNNImputer(n_neighbors=5)
df_knn[features_for_knn.to_list() + ['bmi']] = imputer.fit_transform(df_knn[features_for_knn.to_list() + ['bmi']])

os.makedirs("models", exist_ok=True)      # save trained imputer for future use
joblib.dump(imputer, "models/knn_imputer.pkl")      
print("Imputer saved as models/knn_imputer.pkl")

# create version with KNN imputed BMI
df_imput_knn = df_train.copy()
df_imput_knn['bmi'] = df_knn['bmi']

print("\nMissing values in the train set after imputing:")
print(df_imput_knn['bmi'].isna().sum())

df_imput_knn.to_csv("data/processed/df_imput_knn.csv", index=False)    
print("\nSaved KNN-imputed train dataset as data/processed/df_imput_knn.csv")

plt.figure(figsize=(10,6))
sns.kdeplot(df_train['bmi'], color='red', label='Before Imputation', fill=True, alpha=0.3)
sns.kdeplot(df_imput_knn['bmi'], color='blue', label='After KNN Imputation', fill=True, alpha=0.3)
plt.title("BMI Distribution Before vs After Imputation")
plt.xlabel("BMI")
plt.legend()
plt.show()