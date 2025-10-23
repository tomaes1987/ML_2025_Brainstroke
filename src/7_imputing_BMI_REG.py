import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import os

df_train = pd.read_csv("data/processed/df_train.csv") 

print(f"\nLoaded train set")

df_reg = df_train.copy()
categorical_cols = ['work_type', 'smoking_status']   # encoding (dummy variables), same way
df_reg = pd.get_dummies(df_reg, columns=categorical_cols, drop_first=True)

df_missing = df_reg[df_reg['bmi'].isna()]   # separate rows with and without missing BMI
df_not_missing = df_reg[df_reg['bmi'].notna()]

# Train regression model
X_train_reg = df_not_missing.drop(['bmi', 'stroke'], axis=1)
y_train_reg = df_not_missing['bmi']
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

os.makedirs("models", exist_ok=True)      # save trained imputer for future use
joblib.dump(reg_model, "models/regression_imputer.pkl")  
print("Imputer saved as models/regression_imputer.pkl")

X_missing_reg = df_missing.drop(['bmi', 'stroke'], axis=1)  # predict missing BMI
bmi_pred = reg_model.predict(X_missing_reg)

df_reg.loc[df_reg['bmi'].isna(), 'bmi'] = bmi_pred   # fill missing values

# create version with regression imputed BMI
df_imput_reg = df_train.copy()
df_imput_reg['bmi'] = df_reg['bmi']

print("\nMissing values in the train set after imputing:")
print(df_imput_reg['bmi'].isna().sum())

df_imput_reg.to_csv("data/processed/df_imput_reg.csv", index=False)    
print("\nSaved regression-imputed train dataset as data/processed/df_imput_reg.csv")

plt.figure(figsize=(10,5))
sns.kdeplot(df_train['bmi'], color='red', label='Before Imputation', fill=True, alpha=0.3)
sns.kdeplot(df_imput_reg['bmi'], color='green', label='After Regression Imputation', fill=True, alpha=0.3)
plt.title("BMI Distribution: Original vs KNN vs Regression Imputation")
plt.xlabel("BMI")
plt.legend()
plt.show()