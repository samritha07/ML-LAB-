print("S.SAMRITHA 24BAD103")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\fraud_smote.csv")
df = df.dropna()
df.columns = df.columns.str.strip()
print(df.columns)
target_col = df.columns[-1]
print(df[target_col].value_counts())
X = df.drop(target_col, axis=1)
y = df[target_col]
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)
y_prob_before = model_before.predict_proba(X_test)[:, 1]

acc_before = accuracy_score(y_test, y_pred_before)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(pd.Series(y_res).value_counts())

model_after = LogisticRegression(max_iter=1000)
model_after.fit(X_res, y_res)
y_pred_after = model_after.predict(X_test)
y_prob_after = model_after.predict_proba(X_test)[:, 1]

acc_after = accuracy_score(y_test, y_pred_after)
print("Accuracy Before SMOTE:", acc_before)
print("Accuracy After SMOTE:", acc_after)
plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure()
pd.Series(y_res).value_counts().plot(kind='bar')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
precision_before, recall_before, _ = precision_recall_curve(y_test, y_prob_before)
precision_after, recall_after, _ = precision_recall_curve(y_test, y_prob_after)

auc_before = auc(recall_before, precision_before)
auc_after = auc(recall_after, precision_after)
plt.figure()
plt.plot(recall_before, precision_before, label="Before SMOTE AUC=" + str(round(auc_before, 2)))
plt.plot(recall_after, precision_after, label="After SMOTE AUC=" + str(round(auc_after, 2)))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()




