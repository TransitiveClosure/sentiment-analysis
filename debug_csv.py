import os

import pandas as pd

DATA_PATH = os.path.join('data', 'mock_train.csv')
print(f"Reading from: {DATA_PATH}")

if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
        print("Columns:", df.columns.tolist())
        print("Head:", df.head())
        if 'label' in df.columns:
            print("Label column found.")
        else:
            print("Label column NOT found.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("File not found.")
