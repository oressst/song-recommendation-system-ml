import pandas as pd

df = pd.read_csv('data/dataset.csv')
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("First 5 rows: ", df.head())
print("Last 5 rows: ", df.tail())
