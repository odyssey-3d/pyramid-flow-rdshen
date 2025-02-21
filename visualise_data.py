import pandas as pd

# Load the first parquet file
df1 = pd.read_parquet('/mnt/ssd/datasets/relaion2B-en-research-safe/part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet')

# Basic inspection of the first file
print("\nFirst Parquet File:")
print("Shape:", df1.shape)
print("\nColumns:", df1.columns.tolist())
print("\nData Types:\n", df1.dtypes)
print("\nFirst few rows:\n", df1.head())

# Basic statistics for numerical columns
print("\nSummary statistics for first file:\n", df1.describe())