import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import os

df_train_encoded = pd.read_csv("data/processed/df_train_encoded.csv")

print(f"\nLoaded train encoded set")

df_train_reg = df_train_encoded.copy()

# Train regression model
features = df_train_reg.columns.drop(['stroke','bmi'])
X_train_reg = df_train_reg[features]
y_train_reg = df_train_reg['bmi'].copy()
mask_missing = y_train_reg.isna()

reg_model = LinearRegression()
reg_model.fit(X_train_reg[~mask_missing], y_train_reg[~mask_missing])

os.makedirs("models", exist_ok=True)      # save trained imputer for future use
joblib.dump(reg_model, "models/regression_imputer.pkl")  
print("Imputer saved as models/regression_imputer.pkl")

df_train_reg.loc[mask_missing, 'bmi'] = reg_model.predict(X_train_reg[mask_missing])   # fill missing values

# create version with regression imputed BMI
df_imput_reg = df_train_reg.copy()

print("\nMissing values in the train set after imputing:")
print(df_imput_reg['bmi'].isna().sum())

df_imput_reg.to_csv("data/processed/df_imput_reg.csv", index=False)    
print("\nSaved regression-imputed train dataset as data/processed/df_imput_reg.csv")

plt.figure(figsize=(10,5))
sns.kdeplot(df_train_encoded['bmi'], color='red', label='Before Imputation', fill=True, alpha=0.3)
sns.kdeplot(df_imput_reg['bmi'], color='green', label='After Regression Imputation', fill=True, alpha=0.3)
plt.title("BMI Distribution: Original vs KNN vs Regression Imputation")
plt.xlabel("BMI")
plt.legend()
plt.show()