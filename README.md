# 🧠 **Stroke Prediction Project (Machine Learning)**

This project explores a healthcare dataset to predict the risk of stroke using exploratory data analysis (EDA) and machine learning techniques.

---

## **Project Structure**

```
PROJECT_ML_2025/
│
├── analysis/                          
├── data/                              # Directory for datasets
│   ├── processed/                     # Cleaned and preprocessed datasets
│   ├── raw/                           # Raw dataset
├── models/                            # Directory for trained models or model-related files
├── notebooks/                         # Jupyter notebooks for exploratory analysis
├── reports/                           # Directory for generated reports (e.g., Sweetviz, visualizations)
├── src/                               # Source code for preprocessing and analysis
├── venv/                              # Virtual environment for Python dependencies
├── .gitignore                         # Git ignore file for excluding unnecessary files
├── README.md                          # Documentation for the project
├── requirements.txt                   # List of required Python libraries

---

## **Setup Environment**
### 1. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # on Windows
source venv/bin/activate    # on Mac/Linux
```
### 2. Install required libraries
```bash
pip install -r requirements.txt
```
### 3. Run the project
#### To execute the analysis (e.g., data cleaning, imputations, and visualizations):
```bash
python analiza.py
```
### 4. Required libraries
#### All dependencies are listed in the requirements.txt file.
#### If you need to generate it again, run the following command in your virtual environment:
```bash
pip freeze > requirements.txt
```
#### Example content:
```
pandas==2.3.3
numpy==1.26.4
scikit-learn==1.7.2
seaborn==0.13.2
matplotlib==3.10.7
sweetviz==2.3.1
```
## **Data Preprocessing Summary**
#### Main preprocessing steps performed in analiza.py:
#### - Dropping irrelevant columns (id, Residence_type, gender)
#### - Checking and imputing missing values (especially bmi)
#### - Encoding categorical variables
#### - Using KNN Imputer with numerical + encoded categorical features for more accurate bmi estimation
#### - Verifying imputation quality visually (before/after distribution comparison)

## **EDA (Exploratory Data Analysis)**
#### Conducted using Sweetviz for automated visualization report
#### Additional plots created using Matplotlib and Seaborn
#### Correlation heatmaps and distribution checks were used to guide feature selection

## **Author**
#### Magdalena Melaniuk, Tomasz Bartkowski
#### Machine Learning 2nd semester — project: Stroke Prediction

## **Notes**
#### Ensure Python 3.11+ is installed
#### Always activate the virtual environment before running scripts
#### If any library fails to install due to SSL or network issues, use the --trusted-host flag:
```bash
pip install seaborn --trusted-host pypi.org --trusted-host files.pythonhosted.org
```