print("S.SAMRITHA 24BAD103")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\income_random_forest.csv")

df = df.dropna()

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Income", axis=1)
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_range = [10, 50, 100, 150, 200]
accuracies = []

for n in tree_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

best_n = tree_range[accuracies.index(max(accuracies))]

rf_best = RandomForestClassifier(n_estimators=best_n, random_state=42)
rf_best.fit(X_train, y_train)

y_pred_best = rf_best.predict(X_test)
final_acc = accuracy_score(y_test, y_pred_best)

print("Best Number of Trees:", best_n)
print("Final Accuracy:", final_acc)

plt.figure()
plt.plot(tree_range, accuracies, marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees")
plt.show()

importance = rf_best.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.show()
