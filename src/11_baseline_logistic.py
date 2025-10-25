import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib

df_train_imputed = pd.read_csv("data/processed/df_train_imputed.csv")
df_test_imputed = pd.read_csv("data/processed/df_test_imputed.csv")

# X, y
X_train = df_train_imputed.drop('stroke', axis=1)
y_train = df_train_imputed["stroke"]
X_test = df_test_imputed.drop('stroke', axis=1)
y_test = df_test_imputed["stroke"]

# scaling of numeric variables
num_cols = ['age', 'bmi', 'avg_glucose_level']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# internal validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# model training
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_tr, y_tr)

# evaluation
y_pred = log_model.predict(X_val)
y_prob = log_model.predict_proba(X_val)[:, 1]

print("Baseline Logistic Regression")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


# Baseline Logistic Regression, class weight balanced
log_model_balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_model_balanced.fit(X_tr, y_tr)

# evaluation
y_pred_bal = log_model_balanced.predict(X_val)
y_proba_bal = log_model_balanced.predict_proba(X_val)[:, 1]

print("Baseline Logistic Regression with class_weight='balanced'")
print(f"Accuracy: {accuracy_score(y_val, y_pred_bal):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_proba_bal):.4f}")
print(confusion_matrix(y_val, y_pred_bal))
print(classification_report(y_val, y_pred_bal))    # better results - oversampling SMOTE?

joblib.dump(log_model, "models/logistic_model.pkl")     # models and scaler saved
joblib.dump(log_model_balanced, "models/logistic_balanced.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Baseline Logistic Regression:
# The baseline logistic regression without class balancing predicted all samples as the majority class, 
# giving very high accuracy but 0 recall for stroke.
# Adding class_weight='balanced' improved detection of the minority class (recall 0.82), but overall accuracy dropped (73%).
# ROC-AUC remains similar (~0.85), indicating the model can still rank high-risk cases reasonably well.
# Conclusion: balancing the classes helps logistic regression detect rare events, at the cost of overall accuracy.
