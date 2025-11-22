import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from feature_utils import kdeplot, generate_sweetviz_report, cramers_v_corrected, all_feature_pr_auc, full_model_feature_importance
import matplotlib.pyplot as plt

#IMPUTE MISSING VALUES USING KNN IMPUTER AFRER ENCODING AND SCALING
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

print("\nMissing values in the train set")
print(X_train.isnull().sum())

X_train_check = X_train.copy()
X_test_check = X_test.copy()

X_train_knn = X_train.copy()
X_test_knn = X_test.copy()

# segregation of columns according types
cat_cols = X_train_knn.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_knn.select_dtypes(include=['float64']).columns.tolist()
bin_cols = X_train_knn.select_dtypes(include=['int64']).columns.tolist()

# encode categorical variables ONE-HOT
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# fit and transform train, transform test
train_ohe = ohe.fit_transform(X_train_knn[cat_cols])
test_ohe = ohe.transform(X_test_knn[cat_cols])

# get new column names after encoding
ohe_cols = ohe.get_feature_names_out(cat_cols)

# create DataFrames with encoded categorical variables
X_train_ohe = pd.DataFrame(train_ohe, columns=ohe_cols, index=X_train_knn.index)
X_test_ohe = pd.DataFrame(test_ohe, columns=ohe_cols, index=X_test_knn.index)

# Standard Scaler of numerical variables

scaler = StandardScaler()
bmi_col = "bmi"

X_train_scaled = X_train_knn.copy()
X_test_scaled = X_test_knn.copy()

# scale only numerical columns
X_train_scaled[num_cols] = scaler.fit_transform(X_train_knn[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test_knn[num_cols])

# concatenate encoded categorical variables with scaled numerical and binary variables
X_train_final = pd.concat([X_train_scaled[num_cols + bin_cols], X_train_ohe], axis=1)
X_test_final = pd.concat([X_test_scaled[num_cols + bin_cols], X_test_ohe], axis=1)

# KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_final), columns=X_train_final.columns, index=X_train_final.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns, index=X_test_final.index)

print("\nMissing values in the train set after imputing:")
print(X_train_imputed.isnull().sum())

# inverse scaling for BMI column to original scale
X_train_imputed[num_cols] = scaler.inverse_transform(X_train_imputed[num_cols])
X_test_imputed[num_cols] = scaler.inverse_transform(X_test_imputed[num_cols])


# replace original bmi column in X_train and X_test with imputed values
X_train['bmi'] = X_train_imputed['bmi']
X_test['bmi'] = X_test_imputed['bmi']

# check if the original dataframe and imputed are the same after replacing the bmi column

train_test_idx_equal = (X_train.index == X_train_imputed.index).all(), (X_test.index == X_test_imputed.index).all()
train_test_rows_equal = (X_train_check.drop(columns=['bmi']) == X_train.drop(columns=['bmi'])).all().all(), (X_test_check.drop(columns=['bmi']) == X_test.drop(columns=['bmi'])).all().all()

print(f"\nIndexes equal after imputing: {train_test_idx_equal}")
print(f"Rows equal after imputing (excluding 'bmi' column): {train_test_rows_equal}")

print("\nMissing values in the train set after imputing:")
print(X_train['bmi'].isna().sum())

# plot BMI distribution before and after imputation

kdeplot(X_train_check['bmi'], X_train['bmi'], column_name="bmi", label_before='Before Imputation', label_after='After KNN Imputation')
#kdeplot function was updated to not save the figure automatically
#saved figure manually here
plt.savefig("figures/bmi_before_after_imputation.png", dpi=300, bbox_inches='tight')
plt.close()



# binary columes were not encoded, because they represent numerical values 0 and 1, so yes and no 
# the goal was to take imputed column and replace it in the original dataframes, bacause we are going to make catboost model later and it does not require one-hot encoding or scaling


#DEFINE CLINICAL BINS TO FIND THE TRUE IMPACT ON STROKE (CRAMÉR'S V AND PR-AUC WILL BE CALCULATED LATER IN EDA SCRIPT)
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# clinical bins
bin_specification = {
    'bmi': {
        'bins': [0, 18.5, 24.9, 29.9, 100],
        'labels': ['underweight', 'normal weight', 'overweight', 'obesity']
    },
    'avg_glucose_level': {
        'bins': [0, 99, 125, 200, 1000],
        'labels': ['normal', 'prediabetes', 'diabetes', 'severe diabetes']
    },
    'age': {
        'bins': [0, 18, 35, 50, 65, 100],
        'labels': ['child', 'youth', 'adult', 'middle-aged', 'senior']
    }
}

# apply the same bins to test set
for col, spec in bin_specification.items():
    new_col = col + '_clinical_bin'
    X_train[new_col] = pd.cut(X_train[col], bins=spec['bins'], labels=spec['labels'])
    X_test[new_col] = pd.cut(X_test[col], bins=spec['bins'], labels=spec['labels'])
    print(X_train[[col, new_col]].head())
   
#------------------------------------------------------------------------
# drop original columns after binning
#------------------------------------------------------------------------

drop_columns = ['bmi', 'avg_glucose_level', 'age']

X_train = X_train.drop(columns=drop_columns)
X_test = X_test.drop(columns=drop_columns)

#------------------------------------------------------------------------
#save updated dataframe
#------------------------------------------------------------------------

# save final datasets with imputed BMI and binned features
X_train.to_csv("data/processed/X_train_final.csv", index=False)
X_test.to_csv("data/processed/X_test_final.csv", index=False)

#GENERATE SWEETVIZ REPORT FOR PREPROCESSED DATA
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
generate_sweetviz_report(X_train, output_path="reports/sweetviz_report_after_preprocessing.html")



