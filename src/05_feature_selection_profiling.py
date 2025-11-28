import pandas as pd 
from feature_utils import cramers_v_corrected, all_feature_pr_auc, full_model_feature_importance
from ydata_profiling import ProfileReport
import webbrowser
from pathlib import Path

"""
================================================================================
Analiza wyników feature selection – komentarz dla projektu
================================================================================

1. Cramér’s V – zależność statystyczna
--------------------------------------------------------------------------------
- Cramér’s V pokazuje, jak mocno pojedyncza zmienna jest powiązana z 'stroke' 
  (miara siły asocjacji kategorialnej). Skala 0-1: 0 = brak zależności, 1 = pełna zależność.
- To **tylko korelacja**, nie predykcja (statystyczna miara powiązania, nie ocena klasyfikatora).
- Przykład: 'age' może mieć wysoki Cramér’s V, bo starsi ludzie częściej mają udar, 
  ale sam wiek niekoniecznie rozdzieli przypadki idealnie.

2. PR-AUC i pojedyncze zmienne
--------------------------------------------------------------------------------
- PR-AUC (Precision-Recall Area Under Curve) mierzy zdolność pojedynczej zmiennej
  do rozróżniania przypadków 'stroke' (pozytywne) vs 'no stroke' (negatywne), 
  szczególnie przy nierównomiernym rozkładzie klas (imbalanced dataset).
- **Precision (precyzja)** – odsetek prawdziwych przypadków 'stroke' wśród przewidzianych jako 'stroke' (miara pozytywnej predyktywnej wartości, PPV).  
- **Recall (czułość, sensitivity)** – odsetek wykrytych rzeczywistych przypadków 'stroke' (True Positive Rate, TPR).  
- **Accuracy (dokładność)** – odsetek wszystkich poprawnych przewidywań (zarówno 'stroke' jak i 'no stroke').

- Macierz błędów (confusion matrix) pomaga to zobrazować:

      Prawdziwe:       Predykcja:
                       stroke  no stroke
      stroke           8       2
      no stroke        3       87

  Obliczenia:
  - Precision = 8 / (8+3) ≈ 0.73
  - Recall = 8 / (8+2) = 0.8
  - Accuracy = (8+87)/100 = 0.95

- PR-AUC (pole pod krzywą precision-recall) łączy precision i recall przy różnych progach decyzyjnych,
  pokazując ogólną jakość predykcyjną zmiennej w klasyfikacji 'stroke'.

3. Co ważne w medycznej klasyfikacji
--------------------------------------------------------------------------------
- W przypadku przewidywania udaru, **ważne jest, jakie błędy są krytyczne** (decision cost).
- Jeśli chcemy unikać fałszywych alarmów (false positives), **precision priorytetem**.  
- Recall (czułość) też się liczy, bo nie chcemy przeoczyć prawdziwego udaru, ale można świadomie zaakceptować niższy recall, jeśli zależy nam na ograniczeniu FP.
- PR-AUC jest przydatne, bo syntetycznie pokazuje, jak zmienna balansuje precision i recall (performance measure dla klasy rzadkiej, imbalanced).

4. Feature importance w LightGBM
--------------------------------------------------------------------------------
- Modele drzewiaste patrzą na wszystkie zmienne razem (multivariate analysis).  
- Jeśli inna zmienna dobrze wyjaśnia ryzyko udaru, informacja z 'age' może być „przejęta” przez tę zmienną.  
- Dlatego feature importance 'age' w modelu może być niższa niż w testach univariate.  
- W skrócie: pojedyncza zmienna może wyglądać ważnie statystycznie, ale w modelu jej znaczenie zależy od innych cech (conditional importance).

5. Kluczowe wnioski
--------------------------------------------------------------------------------
- Cramér’s V i PR-AUC dla pojedynczych zmiennych służą do szybkiej selekcji zmiennych i zrozumienia korelacji (exploratory analysis).  
- Feature importance z LightGBM pokazuje, które zmienne naprawdę pomagają modelowi (multivariate predictive importance).  
================================================================================
"""

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

# Comment and Interpretation
# High correlation: 
# The correlations observed are expected given the nature of the dataset. 
# For example, age and ever_married are logically connected, and some work_type categories are mutually exclusive.
# These correlations do not indicate data quality issues but rather inherent relationships in the data.

# Imbalance: 
# Some features, including hypertension and heart_disease show class imbalance.
# This reflects real-world distribution in the dataset and is not a data quality issue. 
# Catboost and other algorithms can handle such imbalances effectively.

# Differences in data types between train and test reports:
# They are no differences in the data types between train and test reports.

# Conclusions
# Single correlation and PR-AUC analyses help identify potentailly useful features,
# hoewever, true feature importance is determined in a multivariate context using models like LightGBM.
# The relative low number of features in the train set, along with that all of them 
# has some predicitive power, make that all features should be retained for modeling.

# The dataset is ready for modeling:
# No missing values remain.
# Categorical variables does not have to be encoded, as catboost can handle category types directly
# and make it more efficient.
# There are no numerical features, because catboost is better with categorical features.
# Catboost handles also non-linear relationships and feature interactions, so correlations and class imbalance
# Catboost also hanldes missing values internally, but in our case there are no missing values.

