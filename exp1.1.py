# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 1.1\data.csv", encoding="ISO-8859-1")
print("S.SAMRITHA 24BAD103")
# Inspect the dataset
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
# Create Sales column
df["Sales"] = df["Quantity"] * df["UnitPrice"]
# Group sales by product (Top 10 products)
product_sales = (
    df.groupby("Description")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
# Bar chart for top 10 products by sales
plt.figure(figsize=(10, 5))
product_sales.plot(kind="bar")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Top 10 Products by Sales")
plt.show()
# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
# Calculate daily sales trend
daily_sales = df.groupby(df["InvoiceDate"].dt.date)["Sales"].sum()
# Line chart for daily sales trend
plt.figure(figsize=(10, 5))
daily_sales.plot()
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Daily Sales Trend")
plt.show()
