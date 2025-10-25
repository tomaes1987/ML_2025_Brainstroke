import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import joblib
import os

df_train = pd.read_csv("data/processed/df_train.csv")
df_test = pd.read_csv("data/processed/df_test.csv")   # only for encoding

print(f"\nLoaded train set - and test set only for encoding")

print("\nMissing values in the train set before imputing:")
print(df_train.isnull().sum())

df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

excluded_cols = ['stroke', 'bmi']
categorical_cols = ['work_type', 'smoking_status']    

combined = pd.concat([df_train_copy, df_test_copy], axis=0, ignore_index=True)  # combining the train and test sets for common encoding

combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)  # encoding (dummy variables) for the whole dataset

df_train_encoded = combined.iloc[:len(df_train_copy), :].copy()    # splitting again 
df_test_encoded = combined.iloc[len(df_train_copy):, :].copy()


df_train_encoded.to_csv("data/processed/df_train_encoded.csv", index=False)
df_test_encoded.to_csv("data/processed/df_test_encoded.csv", index=False)    
print("\nSaved encoded train and test sets")

# KNN Imputation
df_train_knn = df_train_encoded.copy()
features_for_knn = df_train_knn.columns.drop(excluded_cols)

imputer = KNNImputer(n_neighbors=5)
df_train_knn[features_for_knn.to_list() + ['bmi']] = imputer.fit_transform(df_train_knn[features_for_knn.to_list() + ['bmi']])

os.makedirs("models", exist_ok=True)      # save trained imputer for future use
joblib.dump(imputer, "models/knn_imputer.pkl")      
print("Imputer saved as models/knn_imputer.pkl")

# create version with KNN imputed BMI
df_imput_knn = df_train_encoded.copy()
df_imput_knn['bmi'] = df_train_knn['bmi']

print("\nMissing values in the train set after imputing:")
print(df_imput_knn['bmi'].isna().sum())

df_imput_knn.to_csv("data/processed/df_imput_knn.csv", index=False)    
print("\nSaved KNN-imputed train dataset as data/processed/df_imput_knn.csv")

plt.figure(figsize=(10,6))
sns.kdeplot(df_train_encoded['bmi'], color='red', label='Before Imputation', fill=True, alpha=0.3)
sns.kdeplot(df_imput_knn['bmi'], color='blue', label='After KNN Imputation', fill=True, alpha=0.3)
plt.title("BMI Distribution Before vs After Imputation")
plt.xlabel("BMI")
plt.legend()
plt.show()