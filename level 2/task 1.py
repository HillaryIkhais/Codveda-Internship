import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("future.no_silent_downcasting", True)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_curve, auc

churn_train = pd.read_csv("/Users/ikhaisoshuare/Downloads/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
churn_test = pd.read_csv("/Users/ikhaisoshuare/Downloads/Data Set For Task/Churn Prdiction Data/churn-bigml-20.csv")

print(churn_test.head())
print(churn_train.head())
print(churn_train.shape)

#CONVERT YES/NO/TRUE/FALSE COLUMNS TO BINARY
nec_cols = ['International plan', 'Voice mail plan', 'Churn']
for col in nec_cols:
    churn_train[col] = churn_train[col].replace({'Yes': 1, 'No': 0, True: 1, False: 0})
    churn_test[col] = churn_test[col].replace({'Yes': 1, 'No': 0, 'Yes': 1, 'No': 0, True: 1, False: 0})

#ONE HOT ENCODE THE STATE COLUMN
churn_train = pd.get_dummies(churn_train, columns=['State'], drop_first=True)
churn_test = pd.get_dummies(churn_test, columns=['State'], drop_first=True)

churn_train, churn_test = churn_train.align(churn_test, join='inner', axis=1)

X_train = churn_train.drop('Churn', axis=1)
y_train = churn_train['Churn'].astype(int)

X_test = churn_test.drop('Churn', axis=1)
y_test = churn_test['Churn'].astype(int)

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

#LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=7000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Model performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall score {recall_score(y_test,y_pred):.4f}")

print("Top 5")