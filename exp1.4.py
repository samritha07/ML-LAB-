# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
# Load the TAB-separated dataset (IMPORTANT FIX)
df = pd.read_csv(
    r"C:\Users\Lenovo\Downloads\Exp 1.4\marketing_campaign.csv",
    sep="\t"
)
print("S.SAMRITHA 24BAD103")
print("Dataset Columns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
# Check Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
# Feature Engineering
# Create Age column from Year_Birth
df["Age"] = 2024 - df["Year_Birth"]
# Calculate Total Spending
df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)
# Visualization
# Bar plot: Age distribution
plt.figure(figsize=(8,5))
df["Age"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.title("Age Distribution of Banking Customers")
plt.show()
# Box plot: Income distribution
plt.figure(figsize=(6,4))
plt.boxplot(df["Income"].dropna(), vert=False)
plt.xlabel("Income")
plt.title("Income Distribution of Customers")
plt.show()
# Box plot: Total Spending
plt.figure(figsize=(6,4))
plt.boxplot(df["Total_Spending"], vert=False)
plt.xlabel("Total Spending")
plt.title("Customer Spending Pattern")
plt.show()
