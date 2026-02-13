import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("S.SAMRITHA 24BAD103")
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 3.2\auto-mpg.csv")
df.replace("?", np.nan, inplace=True)
df["horsepower"] = pd.to_numeric(df["horsepower"])
df.dropna(inplace=True)
X = df[["horsepower"]]
y = df["mpg"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
degrees = [2, 3, 4]
results = []
train_errors = []
test_errors = []
plt.figure(figsize=(10, 6))
X_sorted = np.sort(X_scaled, axis=0)
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    results.append([d, mse, rmse, r2])
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mse)
    X_sorted_poly = poly.transform(X_sorted)
    y_curve = model.predict(X_sorted_poly)
    plt.plot(X_sorted, y_curve, label=f"Degree {d}")
plt.scatter(X_scaled, y, alpha=0.4, label="Actual Data")
plt.title("Polynomial Curve Fitting for Degrees 2, 3, 4")
plt.xlabel("Scaled Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()
results_df = pd.DataFrame(results, columns=["Degree", "MSE", "RMSE", "R2 Score"])
print("\nPolynomial Regression Model Performance:\n")
print(results_df)
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, test_errors, marker="o", label="Testing Error")
plt.title("Training vs Testing Error Comparison")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
degrees_demo = [1, 2, 3, 6, 10]
plt.figure(figsize=(12, 7))
for d in degrees_demo:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_scaled)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_sorted_poly = poly.transform(X_sorted)
    y_curve = model.predict(X_sorted_poly)
    plt.plot(X_sorted, y_curve, label=f"Degree {d}")
plt.scatter(X_scaled, y, alpha=0.4, label="Actual Data")
plt.title("Underfitting vs Overfitting Demonstration")
plt.xlabel("Scaled Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()
ridge_results = []
alpha_value = 1.0
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    ridge_model = Ridge(alpha=alpha_value)
    ridge_model.fit(X_train_poly, y_train)
    y_pred_ridge = ridge_model.predict(X_test_poly)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mse_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    ridge_results.append([d, mse_ridge, rmse_ridge, r2_ridge])
ridge_df = pd.DataFrame(
    ridge_results, columns=["Degree", "MSE", "RMSE", "R2 Score"]
)
print("\nRidge Regression Performance (Overfitting Control):\n")
print(ridge_df)
