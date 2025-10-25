import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib

df_train_imputed = pd.read_csv("data/processed/df_train_imputed.csv")
X = df_train_imputed.drop('stroke', axis=1)
y = df_train_imputed['stroke']

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# without scaling
models = {
    "xgboost": XGBClassifier(eval_metric='logloss', random_state=42),
    "catboost": CatBoostClassifier(verbose=0, random_state=42)
}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}, ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    joblib.dump(model, f"models/{name}_model.pkl")

# boosting (especially CatBoost) improves both the overall performance and ROC-AUC, which is crucial when classes are uneven

# XGBoost, class balancing
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=(y_tr.value_counts()[0] / y_tr.value_counts()[1])
)
xgb_model.fit(X_tr, y_tr)

y_pred_xgb = xgb_model.predict(X_val)
y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

print("\n=== XGBoost ===")
print(f"Accuracy: {accuracy_score(y_val, y_pred_xgb):.4f}, ROC-AUC: {roc_auc_score(y_val, y_proba_xgb):.4f}")
print(confusion_matrix(y_val, y_pred_xgb))
print(classification_report(y_val, y_pred_xgb))

joblib.dump(xgb_model, "models/xgboost_balanced.pkl")


# CatBoost, class balancing
cat_model = CatBoostClassifier(
    random_state=42,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    class_weights=[1, (y_tr == 0).sum() / (y_tr == 1).sum()]  # balancing classes
)
cat_model.fit(X_tr, y_tr)

y_pred_cat = cat_model.predict(X_val)
y_proba_cat = cat_model.predict_proba(X_val)[:, 1]

print("\n=== CatBoost ===")
print(f"Accuracy: {accuracy_score(y_val, y_pred_cat):.4f}, ROC-AUC: {roc_auc_score(y_val, y_proba_cat):.4f}")
print(confusion_matrix(y_val, y_pred_cat))
print(classification_report(y_val, y_pred_cat))

cat_model.save_model("models/catboost_balanced.cbm")

