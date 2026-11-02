
import pandas as pd

# Membaca dataset SLR dan MLR
slr = pd.read_csv("app/data/data_penjualan.csv")
mlr = pd.read_csv("app/data/data_listrik.csv")

print("=" * 50)
print("DATASET SIMPLE LINEAR REGRESSION")
print("=" * 50)

print("Ukuran dataset:", slr.shape)
print("Kolom:", slr.columns.tolist())
print(slr.head())

print("\nStatistik deskriptif:")
print(slr.describe())

print("\n" + "=" * 50)
print("DATASET MULTIPLE LINEAR REGRESSION")
print("=" * 50)

print("Ukuran dataset:", mlr.shape)
print("Kolom:", mlr.columns.tolist())
print(mlr.head())

print("\nStatistik deskriptif:")
print(mlr.describe())
