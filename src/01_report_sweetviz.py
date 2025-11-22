import pandas as pd
import sweetviz as sv
from feature_utils import load_and_save_csv, generate_sweetviz_report

#DATA LOADING
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
df_raw = load_and_save_csv() 
print(df_raw.dtypes)

print("CSV loaded, number of rows:", len(df_raw))

#GENERATE SWEETVIZ REPORT FOR RAW DATA
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
generate_sweetviz_report(df_raw)

#DATA PREPROCESSING AFTER SWEETVIZ REPORT ANALYSIS
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

df_raport = df_raw.copy()
# check if duplicates
if df_raport.drop(columns=['id']).duplicated().any():
    print("Warning: Duplicates found in the dataset without 'id' column.")
    print("Number of duplicate rows (without 'id'):", df_raport.drop(columns=['id']).duplicated().sum())
elif df_raport.duplicated().any():
    print("Warning: Duplicates found in the dataset.")
    print("Number of duplicate rows:", df_raport.duplicated().sum())
else: 
    print("No duplicates found in the dataset")

# remove 'id' column
df_raport = df_raport.drop(columns=['id'])
# check if dropped
if 'id' in df_raport.columns:
    print("Error: 'id' column was not dropped.")
else:
    print("'id' column successfully dropped.")
