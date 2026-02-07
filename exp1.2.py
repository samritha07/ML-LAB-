# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 1.2\diabetes.csv")
print("S.SAMRITHA 24BAD103")
# Display first and last few rows
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
# Dataset information
print("\nDataset Information:")
print(df.info())
# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
# Replace zero values with NaN for medical columns (as zeros indicate missing data)
columns_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[columns_with_zero] = df[columns_with_zero].replace(0, pd.NA)
# Check missing values after replacement
print("\nMissing Values After Handling Zeros:")
print(df.isnull().sum())
# Histogram for Glucose levels
plt.figure(figsize=(8,5))
plt.hist(df["Glucose"].dropna(), bins=20)
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.title("Distribution of Glucose Levels")
plt.show()
# Boxplot for Glucose levels
plt.figure(figsize=(6,4))
plt.boxplot(df["Glucose"].dropna(), vert=False)
plt.xlabel("Glucose Level")
plt.title("Boxplot of Glucose Levels")
plt.show()
# Histogram for Age distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Patients")
plt.show()
# Boxplot for Age
plt.figure(figsize=(6,4))
plt.boxplot(df["Age"], vert=False)
plt.xlabel("Age")
plt.title("Boxplot of Age Distribution")
plt.show()
