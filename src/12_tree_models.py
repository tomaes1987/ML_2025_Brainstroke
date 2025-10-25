import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib

df_train_imputed = pd.read_csv("data/processed/df_train_imputed.csv")
X = df_train_imputed.drop('stroke', axis=1)
y = df_train_imputed['stroke']

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# without scaling
models = {
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}, ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    joblib.dump(model, f"models/{name}_model.pkl")


# Decision Tree, class weight balanced
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_tr, y_tr)
y_pred_dt = dt_model.predict(X_val)
y_proba_dt = dt_model.predict_proba(X_val)[:, 1]

print("\nDecision Tree with class_weight='balanced'")
print(f"Accuracy: {accuracy_score(y_val, y_pred_dt):.4f}, ROC-AUC: {roc_auc_score(y_val, y_proba_dt):.4f}")
print(confusion_matrix(y_val, y_pred_dt))
print(classification_report(y_val, y_pred_dt))

# Random Forest, class weight balanced
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
rf_model.fit(X_tr, y_tr)
y_pred_rf = rf_model.predict(X_val)
y_proba_rf = rf_model.predict_proba(X_val)[:, 1]

print("\nRandom Forest with class_weight='balanced'")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}, ROC-AUC: {roc_auc_score(y_val, y_proba_rf):.4f}")
print(confusion_matrix(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))

joblib.dump(dt_model, "models/decision_tree_balanced.pkl")    # models saved
joblib.dump(rf_model, "models/random_forest_balanced.pkl")


# Tree Models (Decision Tree & Random Forest):
# Decision Tree and Random Forest achieve high accuracy overall (~92–95%), 
# but they struggle to correctly predict the minority class (stroke = 1).
# Applying class_weight='balanced' slightly changes predictions, but recall for the minority class remains very low for both models.
# ROC-AUC indicates that Random Forest is better at ranking cases (~0.80), but actual detection of rare events is poor.
# Conclusion: tree-based models serve as a useful baseline, but additional techniques 
# (e.g., boosting, class weighting, sampling) are needed to improve minority class detection.