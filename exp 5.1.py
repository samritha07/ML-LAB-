import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
print("S.SAMRITHA 24BAD103")
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 5.1\breast-cancer.csv")
df = df.drop(columns=["id"], errors='ignore')
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
X = df[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]]
y = df["diagnosis"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
k_values = range(1, 21)
accuracy_scores = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))
plt.plot(k_values, accuracy_scores)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()
misclassified = np.where(y_test != y_pred)
print("Number of misclassified samples:", len(misclassified[0]))
X2 = df[["radius_mean","texture_mean"]]
X2_scaled = scaler.fit_transform(X2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2_scaled, y, test_size=0.2, random_state=42)
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train2, y_train2)
x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)
Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y, edgecolor='k')
plt.title("Decision Boundary (K=5)")
plt.show()
