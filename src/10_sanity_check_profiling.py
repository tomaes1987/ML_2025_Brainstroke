import pandas as pd
from ydata_profiling import ProfileReport
import webbrowser
from pathlib import Path

train = pd.read_csv("data/processed/df_train_imputed.csv")
test = pd.read_csv("data/processed/df_test_imputed.csv")

print("TRAIN shape:", train.shape)               # sanity check – quick info
print("TEST shape:", test.shape)
print("\nMissing values:\n", train.isnull().sum())

train_cols = set(train.columns)
test_cols = set(test.columns)
print("Columns in train that are not in test:", train_cols - test_cols)
print("Columns in test that are not in train:", test_cols - train_cols)

# profiling after imputation
profile_train = ProfileReport(train, title="Post-Imputation Sanity Check - TRAIN", explorative=True)
profile_train.to_file("reports/train_post_imputation.html")

profile_test = ProfileReport(test, title="Post-Imputation Sanity Check - TEST", explorative=True)
profile_test.to_file("reports/test_post_imputation.html")

train_path = Path("reports/train_post_imputation.html").resolve()
test_path = Path("reports/test_post_imputation.html").resolve()

webbrowser.open(train_path.as_uri())
webbrowser.open(test_path.as_uri())

# Commentary and Interpretation
# High correlation: 
# The correlations observed are expected given the nature of the dataset. 
# For example, age and ever_married are logically connected, and some work_type categories are mutually exclusive, 
# creating high correlation among dummy variables. This is not an error; it is a natural outcome of encoding 
# categorical variables as dummy/binary features. 
# Tree-based models (Decision Tree, Random Forest, XGBoost, CatBoost) can handle correlated features well, 
# so this should not negatively affect model performance.

# Imbalance: 
# Several features, including stroke, heart_disease, and work_type_Never_worked, show class imbalance.
# This reflects real-world distribution in the dataset and is not a data quality issue. 
# Techniques such as class weighting, resampling, or evaluation metrics like ROC-AUC can handle imbalance during modeling.

# Differences in data types between train and test reports:
# Some dummy variables appear as Boolean in the train report and as Categorical in the test report.
# This is due to Pandas interpreting 0/1 columns differently (int64 vs bool) after reindexing test columns to match train.
# For ML purposes, both bool and int are perfectly valid, as scikit-learn interprets them as numeric features. 
# The difference only affects reporting, not model training.

# Conclusion
# The dataset is ready for modeling:
# No missing values remain.
# All categorical variables have been properly encoded.
# Numerical features are correctly typed.
# Correlations and class imbalance are understood and can be addressed during modeling if needed.
# Minor differences in how dummy variables are displayed between train and test do not impact machine learning models.