import pandas as pd
from sklearn.impute import KNNImputer
import joblib
import os

imputer = joblib.load("models/knn_imputer.pkl")    # load the trained KNN imputer

df_train_encoded = pd.read_csv("data/processed/df_train_encoded.csv")
df_test_encoded = pd.read_csv("data/processed/df_test_encoded.csv")

df_train_knn = df_train_encoded.copy()
df_test_knn = df_test_encoded.copy()

excluded_cols = ['stroke', 'bmi']    # apply the same encoding as for train

features_for_knn = df_train_knn.columns.drop(excluded_cols)                   # ensuring there are exactly the same columns in train and test set

df_test_knn[features_for_knn.to_list() + ['bmi']] = imputer.transform(        # appling transformation (without fitting again)
    df_test_knn[features_for_knn.to_list() + ['bmi']]
)

print("Missing values in test set after KNN imputation:", df_test_knn['bmi'].isna().sum())

df_test_imputed = df_test_knn.copy()

df_test_imputed.to_csv("data/processed/df_test_imputed.csv", index=False)    
print("\nSaved KNN-imputed test dataset as data/processed/df_test_imputed.csv")
