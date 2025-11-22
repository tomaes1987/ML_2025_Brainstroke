import pandas as pd
import numpy as np
import os
import sweetviz as sv
import requests
from io import StringIO
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

def generate_sweetviz_report(df, output_path="reports/report_sweetviz.html"):
    """
    Generates a Sweetviz EDA report and saves it as HTML.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        output_path (str): Path where the HTML report will be saved
                          (default: "reports/report_sweetviz.html")
    
    Returns:
        None
    """
    # Extract folder path from output_path
    folder = os.path.dirname(output_path)
    
    # Create folder if it doesn't exist
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created")
    
    # Generate and save report
    print("Generating Sweetviz report...")
    report = sv.analyze(df)
    report.show_html(output_path)
    print(f"Report saved to: {output_path}")

# -----------------------------
def load_and_save_csv(url=None, save_folder='data/raw'):
    """
    Loads CSV from a link and saves it to 'data/raw' folder.
    If URL is not provided, asks the user to paste it.
    
    Args:
        url (str, optional): The link to CSV file. Default is None.
        save_folder (str): Folder to save CSV (default 'data/raw').
    
    Returns:
        pd.DataFrame: The loaded dataset or None if error/exit.
    """
    # Ensure the folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # List existing files in the folder
    existing_files = [f for f in os.listdir(save_folder) 
                      if os.path.isfile(os.path.join(save_folder, f))]
    
    if existing_files:
        print("Existing files in the folder:")
        for file in existing_files:
            print(f"- {file}")
    else:
        print(f"No files found in the folder '{save_folder}'.")
    
    # Get a valid URL from the user
    while not url or not url.lower().endswith('.csv'):
        url = input("Provide URL (must end with .csv) or type 'e' to exit: ").strip()
        
        if url.lower() == 'e':
            print("Exiting without downloading.")
            if existing_files:
                df = pd.read_csv(os.path.join(save_folder, existing_files[0]))
                print(df.head())
                return df
            else:
                print("No existing files to load.")
                return None
    
    # Extract the filename and prepare the save path
    filename = url.split("/")[-1]
    save_path = os.path.join(save_folder, filename)
    
    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File '{filename}' already exists. Loading the dataset...")
        try:
            existing_df = pd.read_csv(save_path)
            print("Preview of the existing dataset:")
            print(existing_df.head())
            return existing_df
        except Exception as e:
            print(f"Error reading the existing file: {e}")
            return None
    
    # Download and save the file
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(save_path, index=False)
        print(f"Dataframe saved in {save_path}")
        print("Preview of the dataframe:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error during download or saving: {e}")
        return None

# -----------------------------
def cramers_v_corrected(x, y):
    """
    Cramér's V with bias correction (Bergmsa/Wicher) between two categorical variables.

    Args:
        x, y: categorical variables
    Returns:
        float: corrected Cramér's V statistic in a range [0, 1], where 0 means no association and 1 means perfect association
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))


# -----------------------------
def all_feature_pr_auc(df, target, n_splits=5, random_state=42):
    """
    Calculate PR-AUC (Precision-Recall AUC) for each categorical features and target using cross-validated
    LightGBM classifier.
    
    Args:
        df: DataFrame with features and target
        target (str): name of target column
        n_splits: number of CV folds, default 5
        random_state: random seed, default 42

    Returns:
        DataFrame with features sorted by PR-AUC (descending)
    """
    y = df[target].values
    features = df.drop(columns=[target])
    categorical_cols = features.select_dtypes(include="category").columns
    
    if len(categorical_cols) == 0:
        raise ValueError("No categorical columns found in dataset")
    
    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for feature in categorical_cols:
        X = features[[feature]]
        fold_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1  # eliminates warnings
            )
            clf.fit(X.iloc[train_idx], y[train_idx])
            preds = clf.predict_proba(X.iloc[test_idx])[:, 1]
            fold_pr_auc = average_precision_score(y[test_idx], preds)
            fold_scores.append(fold_pr_auc)
        
        mean_pr_auc = np.mean(fold_scores)
        results.append({"feature": feature, "pr_auc": mean_pr_auc})
    
    return pd.DataFrame(results).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

# -----------------------------
def full_model_feature_importance(df, target, n_splits=5, random_state=42):
    """
    Calculates feature importance using all features together with CV.
    Returns both importance scores and model performance.
    
    Args:
        df: DataFrame with features and target
        target: name of target column
        n_splits: number of CV folds, default 5
        random_state: random seed, default 42
    
    Returns:
        dict with 'importance_df', 'mean_pr_auc', 'pr_auc_std', 'fold_scores'
    """
    y = df[target].values
    X = df.drop(columns=[target])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Collect importance and scores from each fold
    importance_list = []
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds = clf.predict_proba(X.iloc[test_idx])[:, 1]
        
        # PR-AUC for this fold
        fold_pr_auc = average_precision_score(y[test_idx], preds)
        fold_scores.append(fold_pr_auc)
        
        # Feature importance for this fold
        importance_list.append(clf.feature_importances_)
    
    # Average importance across all folds
    mean_importance = np.mean(importance_list, axis=0)
    std_importance = np.std(importance_list, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_importance,
        'importance_std': std_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Add percentage contribution
    importance_df['importance_pct'] = (
        100 * importance_df['importance'] / importance_df['importance'].sum()
    )
    
    return {
        'importance_df': importance_df,
        'mean_pr_auc': np.mean(fold_scores),
        'pr_auc_std': np.std(fold_scores),
        'fold_scores': fold_scores
    }
# -----------------------------
def kdeplot(data_before, data_after, column_name, label_before='Before Imputation', label_after='After Imputation'):
    """
    Plots KDE plots for before and after imputation data with a custom color palette.

    Args:
        data_before (pd.Series): Data before imputation
        data_after (pd.Series): Data after imputation
        label_before (str): Label for the data before imputation
        label_after (str): Label for the data after imputation
        column_name (str): Name of the column being plotted
    """
    #custom palette
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data_before, label=label_before, alpha=0.3, color='orange', fill=True)
    sns.kdeplot(data_after, label=label_after, alpha=0.3, color="blue", fill=True)
    plt.title(f"{column_name} distribution Before vs After Imputation")
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.legend()
   
    