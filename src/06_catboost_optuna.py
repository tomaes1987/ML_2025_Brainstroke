"""
==================== Raport Modelu CatBoost (Stroke Prediction) ====================

Cel: Predykcja ryzyka udaru u pacjentów na podstawie danych klinicznych.
Problem: Silnie niezrównoważone klasy (większość pacjentów nie miała udaru).

Metody:
- Model: CatBoostClassifier
- Funkcja celu Optuna: maksymalizacja PR-AUC w cross-validation (10 foldów)
- Parametry CatBoost optymalizowane przez Optuna: iterations, learning_rate, depth, l2_leaf_reg,
  min_data_in_leaf, random_strength, bagging_temperature, subsample, colsample_bylevel
- Eval_metric CatBoost: "Recall" → przyspieszenie nauki klasy 1 i monitorowanie early stopping
- Class weights: zwiększona waga klasy 1 (1.25 × neg/pos), aby podnieść recall kosztem precision
- Threshold decyzyjny: 0.35 dla klasy 1

Wyniki (test set):
- Recall klasy 1: 0.92
- Precision klasy 1: 0.08
- F1 klasy 1: 0.15
- PR-AUC: ~0.237
- Confusion Matrix:
    [[TN, FP],
     [FN, TP]] = [[440, 532],
                   [4, 46]]

Interpretacja:
- Wysoki recall klasy 1 jest priorytetem w medycynie (minimalizacja false negatives).
- Niska precyzja jest akceptowalna w tego typu rzadkich, wysokiego ryzyka zdarzeniach.
- Strategia jest zgodna z literaturą: PR-AUC jako funkcja celu + eval_metric="Recall" + zwiększone class_weights.

Wnioski:
- Model skutecznie wychwytuje pacjentów zagrożonych udarem.
- Możliwa dalsza regulacja precision/recal przez threshold lub drobną zmianę class_weights.
- Wizualizacja feature importance i SHAP pozwala interpretować wpływ cech na predykcje.

===================================================================================
"""

import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix, recall_score, precision_recall_curve
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

#--------------- Założenia modelu ---------------
#--------------------------------------------------------
#--------------------------------------------------------

# Model predykcyjny, który na podstawie danych pacjenta przewidzi ryzyko wystąpienia udaru.
# Jest to klasyczny przypadek w medycynie z niezrównoważonymi danymi — większość pacjentów nie miała udaru.
# Skupiamy się na metrykach istotnych dla klasy mniejszościowej (pacjenci z udarem).
# ponieważ w medycynie istotniejsze jest wychwycenie wszystkich przypadków ryzyka niż minimalizacja fałszywych alarmów.
# W tym celu zastosujemy CatBoost z ustawionymi wagami klas (class_weights), co zazwyczaj daje lepsze wyniki niż regresja logistyczna
# przy silnie niezrównoważonych danych.
# Wyniki oceniamy na podstawie macierzy błędów, classification report oraz PR-AUC.

# Główny model oparty na CatBoost nie wymaga one-hot encodingu ani skalowania cech numerycznych.
# CatBoost jest dobrym punktem wyjścia, ponieważ:
# - dobrze działa na małych i średnich zbiorach danych,
# - pozwala łatwo ustawić wagi klas (class_weights) dla problemów niezrównoważonych,
# - jest odporny na wartości odstające i brakujące dane,
# - radzi sobie z dużą liczbą zmiennych kategorycznych bez dodatkowego przetwarzania,
# - w wielu przypadkach przewyższa XGBoost przy danych kategorycznych i niezrównoważonych.


#--------------- Model assumptions ---------------
#--------------------------------------------------------
#--------------------------------------------------------
# Predictive model to estimate a patient's risk of stroke based on their data.
# This is a classic medical case with imbalanced data — most patients did not experience a stroke.
# We focus on metrics relevant for the minority class (patients with stroke).
# because in healthcare it is more important to detect all at-risk cases than to minimize false alarms.
# To address this, we use CatBoost with class_weights, which usually outperforms logistic regression
# on highly imbalanced datasets.
# We evaluate the results using confusion matrix, classification report, and PR-AUC.

# The main CatBoost model does not require one-hot encoding or scaling of numerical features.
# CatBoost is a good starting point because:
# - it works well on small to medium-sized datasets,
# - allows easy setting of class weights for imbalanced problems,
# - is robust to outliers and missing values,
# - handles a large number of categorical variables without additional preprocessing,
# - in many cases, it outperforms XGBoost on categorical and imbalanced data.

# --------------------------------------------------------
# 1. Data loading and preprocessing
# --------------------------------------------------------
X_train = pd.read_csv("data/processed/X_train_final.csv")
X_test = pd.read_csv("data/processed/X_test_final.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_train = y_train.squeeze()
y_test = y_test.squeeze()
  
# change dtype of categorical features
# --------------------------------------------------------
for dataframe in [X_train, X_test]:
    for col in dataframe.select_dtypes(include=['object']).columns:
        dataframe[col] = dataframe[col].astype('category')
# change dtype of hypertension, heart_diesase
for col in ['hypertension', 'heart_disease']:
    X_train[col] = X_train[col].map({0: 'No', 1: 'Yes'}).astype('category')
    X_test[col] = X_test[col].map({0: 'No', 1: 'Yes'}).astype('category')
    
cat_features = X_train.select_dtypes(include=['category']).columns.tolist()

# compute class weights
# --------------------------------------------------------
neg, pos = y_train.value_counts()
class_weights = [1, 1.25 * neg / pos]
print(f"\nClass weights: {class_weights}")


# --------------------------------------------------------
# 2. Funkcja celu – Optuna maksymalizuje PR-AUC
# 2. Objective function - Optuna maximizes PR-AUC
# --------------------------------------------------------
def objective(trial):

    # Parameters to optimize
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",10, 80),
        "random_strength": trial.suggest_float("random_strength", 1, 20),             
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "eval_metric": "Recall",
        "cat_features": cat_features,
        "random_seed": 42,
        "verbose": 0,
        "class_weights": class_weights
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pr_aucs = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)

        probs = model.predict_proba(X_val)[:, 1]

        pr_auc = average_precision_score(y_val, probs)
        pr_aucs.append(pr_auc)

    return np.mean(pr_aucs)

# --------------------------------------------------------
# 3. Optuna study
# --------------------------------------------------------
study = optuna.create_study(
    direction="maximize",
    sampler = optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50)

best_params = study.best_params

print("\n============================")
print(" Najlepsze parametry Optuna ")
print("============================")
print(best_params)
print(f"\nBest mean PR-AUC: {study.best_value:.4f}")

# --------------------------------------------------------
# 4. Final model training on full training set
# --------------------------------------------------------
print("\nTrening of final model on full training set...")

final_model = CatBoostClassifier(
    **best_params,
    eval_metric="Recall",
    class_weights=class_weights,
    cat_features=cat_features,
    random_seed=42,
    verbose=0
)

final_model.fit(X_train, y_train, cat_features=cat_features)
print("Model trained.")

# --------------------------------------------------------
# 5. Final evaluation on test set
# --------------------------------------------------------

probs_test = final_model.predict_proba(X_test)[:, 1]
y_pred = (probs_test > 0.35).astype(int)

# confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

# PR-AUC
pr_auc = average_precision_score(y_test, probs_test)
print(f"PR-AUC: {pr_auc:.4f}")

# Results for class 1 (stroke)
report = classification_report(y_test, y_pred, output_dict=True)
recall_1 = report["1"]["recall"]
precision_1 = report["1"]["precision"]
f1_1 = report["1"]["f1-score"]
print(f"Class 1 (stroke) - Recall: {recall_1:.4f}, Precision: {precision_1:.4f}, F1-Score: {f1_1:.4f}")

# --------------------------------------------------------
# 6. SHAP for CatBoost
# --------------------------------------------------------
# eplain the model's predictions using SHAP
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values.values, X_test, show=False)
plt.savefig("figures/shap_summary_plot.png", dpi=300, bbox_inches='tight')
            
# Summary plot - global feature importance
shap.summary_plot(shap_values.values, X_test, plot_type="bar", show=False)
plt.savefig("figures/shap_summary_plot_bar.png", dpi=300, bbox_inches='tight')

# Force plot for a single prediction
force_plot = shap.force_plot(
    base_value=shap_values.base_values[0],
    shap_values=shap_values.values[0],
    features=X_test.iloc[0]
)
shap.save_html("figures/shap_force_plot_patient_0.html", force_plot)


# waterfall plot for a single prediction
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("figures/shap_waterfall_plot_patient_0.png", dpi=300, bbox_inches='tight')







