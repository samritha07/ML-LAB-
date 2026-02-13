# MULTILINEAR REGRESSION PROJECT 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
print("S.SAMRITHA 24BAD103")
# Load Dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 3.1\StudentsPerformance.csv")
print("Dataset Loaded Successfully!")
print(df.head())
# Encode categorical features
label_enc = LabelEncoder()
df["parental level of education"] = label_enc.fit_transform(
    df["parental level of education"]
)
df["test preparation course"] = label_enc.fit_transform(
    df["test preparation course"]
)
# Target Variable = Final Exam Score
df["final_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3
# Add extra features (simulated)
np.random.seed(42)

df["study_hours"] = np.random.randint(1, 8, df.shape[0])
df["attendance"] = np.random.randint(60, 100, df.shape[0])
df["sleep_hours"] = np.random.randint(4, 10, df.shape[0])
# Input Features
X = df[
    ["study_hours",
     "attendance",
     "parental level of education",
     "test preparation course",
     "sleep_hours"]
]
# Target
y = df["final_score"]
# Handle Missing Values
X.iloc[10:20, 0] = np.nan
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
print("Missing Values Handled!")
# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train Multilinear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n MODEL PERFORMANCE")
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
# Regression Coefficients
feature_names = [
    "Study Hours",
    "Attendance %",
    "Parental Education",
    "Test Preparation",
    "Sleep Hours"
]
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
})
print("\n Feature Influence:")
print(coefficients)
# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("\nRidge R² Score:", r2_score(y_test, ridge_pred))
# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)
print("Lasso R² Score:", r2_score(y_test, lasso_pred))

# 1. Predicted vs Actual Plot
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Predicted vs Actual Exam Scores")
plt.show()
# 2. Coefficient Magnitude Plot
plt.figure(figsize=(7,5))
plt.bar(feature_names, model.coef_)
plt.xticks(rotation=30)
plt.title("Regression Coefficient Comparison")
plt.show()
# 3. Residual Distribution Plot
residuals = y_test - y_pred
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=20)
plt.title("Residual Distribution Plot")
plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.show()

