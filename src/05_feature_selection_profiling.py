import pandas as pd 
from feature_utils import cramers_v_corrected, all_feature_pr_auc, full_model_feature_importance
from ydata_profiling import ProfileReport
import webbrowser
from pathlib import Path

X_train = pd.read_csv("data/processed/X_train_final.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

X_test = pd.read_csv("data/processed/X_test_final.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

#------------------------------------------------------------------------
# change object dtypes to category for modeling purposes(catboost better handles category types, lightgbm as well)
#------------------------------------------------------------------------

for dataframe in [X_train, X_test]:
    for col in dataframe.select_dtypes(include=['object']).columns:
        dataframe[col] = dataframe[col].astype('category')

# change dtype of hypertension, heart_diesase
for col in ['hypertension', 'heart_disease']:
    X_train[col] = X_train[col].map(
        {0: 'No',
         1: 'Yes'}).astype('category')
    X_test[col] = X_test[col].map(
        {0: 'No',
         1: 'Yes'}).astype('category')

# show updated dtypes
print(X_train.dtypes)
print(y_train.dtypes)
print(X_test.dtypes)
print(y_test.dtypes)

# ensure both train and test have the same categories for each categorical column    
for col in X_train.select_dtypes(include='category').columns:
    X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)

#Cramér's V test with bias correction (statistical association, but not predictive power)
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
features = X_train.copy()
target = y_train.copy().squeeze()

for feature in features.columns:
    cv = cramers_v_corrected(features[feature], target)
    #feature drop based on Cramér's V threshold
    if cv < 0.05:
        # practical threshold for dropping features with very low association
        print(f"Consider feature {feature} drop due to low Cramér's V value: {cv:.4f}")
    else:
        print(f"Cramér's V between {feature} and stroke: {cv:.4f}")

#PR-AUC test for single categorical features
#if PR-AUC is low, feature has low predictive power alone
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
pr_auc_df = all_feature_pr_auc(pd.concat([X_train, y_train], axis=1), 'stroke', n_splits=5, random_state=42)
print(pr_auc_df)

#Full model feature importance based on PR-AUC
#Unique imput of feature in LightGBM model
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
results = full_model_feature_importance(pd.concat([X_train, y_train], axis=1), 'stroke')
print(f"Model PR-AUC: {results['mean_pr_auc']:.4f} ± {results['pr_auc_std']:.4f}")
print("\nFeature Importance:")
print(results['importance_df'])


#SANITY CHECK PROFILING
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# sanity check – quick info
print("TRAIN shape:", X_train.shape)              
print("TEST shape:", X_test.shape)
print("\nMissing values:\n", X_train.isnull().sum())

train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
print("Columns in train that are not in test:", train_cols - test_cols)
print("Columns in test that are not in train:", test_cols - train_cols)

# profiling after imputation
profile_train = ProfileReport(X_train, title="Post-Imputation Sanity Check - TRAIN", explorative=True)
profile_train.to_file("reports/train_post_imputation.html")

profile_test = ProfileReport(X_test, title="Post-Imputation Sanity Check - TEST", explorative=True)
profile_test.to_file("reports/test_post_imputation.html")

train_path = Path("reports/train_post_imputation.html").resolve()
test_path = Path("reports/test_post_imputation.html").resolve()

webbrowser.open(train_path.as_uri())
webbrowser.open(test_path.as_uri())