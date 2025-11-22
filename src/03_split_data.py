import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/df_clean.csv")

target = 'stroke' 
features = [col for col in df.columns if col != target and col != 'id'] 

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train X shape: {X_train.shape}")
print(f"Test X shape: {X_test.shape}")

# stroke distribution comparison
print("\nStroke distribution (%):")   
print("Train:", y_train.mean() * 100)
print("Test :", y_test.mean() * 100)

# confirm column consistency, set is used to ignore order
if set(X_train.columns) == set(X_test.columns):   
    print("Column structure consistent between train and test.")
else:
    print("Column mismatch detected — check data cleaning steps.")

X_train.to_csv("data/processed/X_train.csv", index=False)    
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("\nTrain/test data saved in 'data/processed/' folder")
