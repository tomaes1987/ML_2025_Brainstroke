import pandas as pd
from sklearn.impute import KNNImputer
import joblib
import os

imputer = joblib.load("models/knn_imputer.pkl")    # load the trained KNN imputer

df_train = pd.read_csv("data/processed/df_train.csv")
df_test = pd.read_csv("data/processed/df_test.csv")

df_train_knn = df_train.copy()
df_test_knn = df_test.copy()

excluded_cols = ['stroke', 'bmi']    # apply the same encoding as for train
categorical_cols = ['work_type', 'smoking_status']

df_train_knn = pd.get_dummies(df_train_knn, columns=categorical_cols, drop_first=True)
df_test_knn = pd.get_dummies(df_test_knn, columns=categorical_cols, drop_first=True)

df_test_knn = df_test_knn.reindex(columns=df_train_knn.columns, fill_value=0)    # matching columns in train and test set

features_for_knn = df_train_knn.columns.drop(excluded_cols)                      # ensuring there are exactly the same columns in train and test set

df_test_knn[features_for_knn.to_list() + ['bmi']] = imputer.transform(           # appling transformation (without fitting again)
    df_test_knn[features_for_knn.to_list() + ['bmi']]
)

print("Missing values in test set after KNN imputation:", df_test_knn['bmi'].isna().sum())

df_test_imputed = df_test_knn.copy()

df_test_imputed.to_csv("data/processed/df_test_imputed.csv", index=False)    
print("\nSaved KNN-imputed test dataset as data/processed/df_test_imputed.csv")