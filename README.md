# 🧠 **Stroke Prediction with CatBoost & Optuna**

# PROJECT_ML_2025

Predict stroke risk in patients using clinical data.  
Highly imbalanced dataset — main goal is **high recall** for stroke patients (class 1).

**Model:** CatBoostClassifier  
- Hyperparameter tuning with Optuna (objective: PR-AUC)  
- CatBoost `eval_metric="Recall"` to focus on class 1  
- Class weights: 1.25× for class 1  
- Decision threshold: 0.35  

**Test Set Results (class 1):**  
- Recall: 0.92 | Precision: 0.08 | F1: 0.15 | PR-AUC: 0.2366  
*Results are printed in console; the model itself is not saved.*

**Feature analysis:** SHAP plots for global and individual explanations are saved in `figures/`.  

**Usage:**  
1. Load data: `X_train_final.csv`, `X_test_final.csv`, `y_train.csv`, `y_test.csv`  
2. Run `06_catboost_optuna.py`  
3. Confusion matrices, classification reports, and SHAP figures are saved/printed as described.


## 📁 Project Structure
```
PROJECT_ML_2025/
│
├── data/
│   ├── raw/                          # Original, immutable data
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/                    # Cleaned and split data
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── 01_report_sweetviz.py        # EDA with SweetViz
│   ├── 02_validation_cleaning.py    # Data validation and cleaning
│   ├── 03_split_data.py             # Train/test split with stratification
│   ├── 04_imputing_binning.py       # KNN imputation + binning
│   ├── 05_feature_selection_profiling.py  # Feature engineering and selection, EDA after imputing
│   ├── 06_catboost_optuna.py        # CatBoost training with Optuna tuning, shap 
│   └── feature_utils.py             # Utility functions for feature processing
│
├── notebooks/                        # Jupyter notebooks for exploration
│
├── figures/                          # Generated plots and visualizations
│
├── reports/                          # Generated analysis reports
│
├── catboost_info/                   # CatBoost training logs
│
├── venv/                            # Virtual environment
│
├── .gitignore
└── README.md
```
## 🚀 Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Step 1: EDA
python src/01_report_sweetviz.py

# Step 2: Data validation
python src/02_validation_cleaning.py

# Step 3: Split data
python src/03_split_data.py

# Step 4: Imputation & binning
python src/04_imputing_binning.py

# Step 5: Feature engineering
python src/05_feature_selection_profiling.py

# Step 6: Train model
python src/06_catboost_optuna.py
```

## 📊 Dataset

- **Source**: Healthcare Stroke Dataset
- **Target**: `stroke` (binary classification)
- **Features**: Age, BMI, glucose level, hypertension, heart disease, etc.
- **Class Imbalance**: ~5% stroke cases

## 🛠️ Technologies

- **Python 3.x**
- **scikit-learn** - Data preprocessing, KNN imputation
- **CatBoost** - Gradient boosting classifier
- **Optuna** - Hyperparameter optimization
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **SweetViz** - Automated EDA

## **Author**
#### Magdalena Melaniuk, Tomasz Bartkowski
#### Machine Learning 2nd semester — project: Stroke Prediction


