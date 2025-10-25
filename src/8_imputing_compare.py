import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

df_train_encoded = pd.read_csv("data/processed/df_train_encoded.csv")
df_imput_knn = pd.read_csv("data/processed/df_imput_knn.csv") 
df_imput_reg = pd.read_csv("data/processed/df_imput_reg.csv") 

plt.figure(figsize=(10,6))
sns.kdeplot(df_train_encoded['bmi'], color='red', label='Before Imputation', fill=True, alpha=0.25, linewidth=1.25)   # KDE curves
sns.kdeplot(df_imput_knn['bmi'], color='royalblue', label='KNN Imputation', fill=True, alpha=0.25, linewidth=1.25)
sns.kdeplot(df_imput_reg['bmi'], color='seagreen', label='Regression Imputation', fill=True, alpha=0.25, linewidth=1.25)

plt.axvline(df_train_encoded['bmi'].median(), color='red', linestyle='--', linewidth=1)   # median lines
plt.axvline(df_imput_knn['bmi'].median(), color='royalblue', linestyle='--', linewidth=1)
plt.axvline(df_imput_reg['bmi'].median(), color='seagreen', linestyle='--', linewidth=1)

plt.title("BMI Distribution Comparison: Original vs KNN vs Regression Imputation", fontsize=14, pad=15)
plt.xlabel("BMI", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Dataset Version", fontsize=10)
plt.grid(alpha=0.3)

plt.show()

print("Summary comparison (mean ± std):")
print(pd.DataFrame({
    'Original': [df_train_encoded['bmi'].mean(), df_train_encoded['bmi'].std()],
    'KNN': [df_imput_knn['bmi'].mean(), df_imput_knn['bmi'].std()],
    'Regression': [df_imput_reg['bmi'].mean(), df_imput_reg['bmi'].std()]
}, index=['Mean', 'Std']).round(2))

# Both KNN and regression imputation maintain a very similar BMI structure to the original. 
# KNN appears to be marginally better because the KNN curve better reflects the original distribution and 
# it maintains a larger standard deviation (i.e., smooths the data less), 
# which usually indicates a more realistic representation of the population.

# KS test (Kolmogorov–Smirnov test)
bmi_original = df_train_encoded['bmi'].dropna()  # droping of NaN in order to compare
bmi_knn = df_imput_knn['bmi']
bmi_reg = df_imput_reg['bmi']

ks_knn = ks_2samp(bmi_original, bmi_knn)   # Kolmogorov–Smirnov test
ks_reg = ks_2samp(bmi_original, bmi_reg)

print("\nKNN Imputation vs Original:")
print(f"  KS Statistic = {ks_knn.statistic:.4f}, p-value = {ks_knn.pvalue:.4f}")
print("\nRegression Imputation vs Original:")
print(f"  KS Statistic = {ks_reg.statistic:.4f}, p-value = {ks_reg.pvalue:.4f}")

if ks_knn.pvalue > 0.05:    # interpretation
    print("\nKNN distribution not significantly different from original.")
else:
    print("\nKNN distribution differs significantly from original.")

if ks_reg.pvalue > 0.05:
    print("Regression distribution not significantly different from original.")
else:
    print("Regression distribution differs significantly from original.")

# Both imputation methods maintained the original BMI distribution pattern, with no statistically significant differences (p > 0.05).
# The KNN imputation again showed slightly better alignment with the original data (KS = 0.0099 vs 0.0157), 
# suggesting a more faithful reconstruction of the BMI distribution.

df_train_imputed = df_train_encoded.copy()
df_train_imputed['bmi'] = df_imput_knn['bmi']

df_train_imputed.to_csv("data/processed/df_train_imputed.csv", index=False)    
print("\nSaved BMI imputed train dataset as data/processed/df_train_imputed.csv")
