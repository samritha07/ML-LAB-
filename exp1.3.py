# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 1.3\Housing.csv")
print("S.SAMRITHA 24BAD103")
# Inspect dataset columns
print("Dataset Columns:")
print(df.columns)
# View first 5 rows
print("\nFirst 5 rows:")
print(df.head())
# Dataset information
print("\nDataset Information:")
print(df.info())
# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Detect missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
# Scatter plot: Area vs Price
plt.figure(figsize=(6,4))
plt.scatter(df["area"], df["price"])
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("Area vs House Price")
plt.show()
# Scatter plot: Bedrooms vs Price
plt.figure(figsize=(6,4))
plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")
plt.title("Bedrooms vs House Price")
plt.show()
# Correlation heatmap (numerical columns only)
corr = df.select_dtypes(include=["int64", "float64"]).corr()
plt.figure(figsize=(10,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap of Housing Features")
plt.show()



