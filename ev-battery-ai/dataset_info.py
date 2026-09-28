import pandas as pd

file_path = "dataset/ev battery_failure prediction Dataset.csv"

df = pd.read_csv(file_path)

print("\n--- DATASET BOYUTU ---")
print(df.shape)

print("\n--- SÜTUNLAR ---")
print(df.columns.tolist())

print("\n--- İLK 5 SATIR ---")
print(df.head())

print("\n--- VERİ TİPLERİ ---")
print(df.dtypes)

print("\n--- EKSİK VERİLER ---")
print(df.isnull().sum())

print("\n--- İSTATİSTİKLER ---")
print(df.describe())