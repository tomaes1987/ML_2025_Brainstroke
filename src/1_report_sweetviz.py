import pandas as pd
import sweetviz as sv
import os
from functions.data_loading import load_and_save_csv

df = load_and_save_csv() 

print("CSV loaded, number of rows:", len(df))

if not os.path.exists("reports"):
    os.makedirs("reports")
    print("Reports folder created")
report = sv.analyze(df)
report.show_html("reports/report_sweetviz.html")
print("Report generated")
