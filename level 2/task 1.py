import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("future.no_silent_downcasting", True)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_curve, auc, roc_auc_score, classification_report

churn_train = pd.read_csv("/Users/ikhaisoshuare/Downloads/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
churn_test = pd.read_csv("/Users/ikhaisoshuare/Downloads/Data Set For Task/Churn Prdiction Data/churn-bigml-20.csv")

print(churn_test.head())
print(churn_train.head())
print(churn_train.shape)

churn_train['Total_Usage'] = churn_train['Total day minutes'] + churn_train['Total eve minutes'] + churn_train['Total night minutes']


#CONVERT YES/NO/TRUE/FALSE COLUMNS TO BINARY
cols_to_drop = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
churn_train = churn_train.drop(columns=cols_to_drop)
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
roc_score = roc_auc_score(y_test, y_prob)

print("Model performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall score {recall_score(y_test,y_pred):.4f}")
print(f"AUC Score: {roc_score:.4f}\n")
print(classification_report(y_test, y_pred))

print("Top 5 business logic translators")
coefficients = model.coef_[0]
feat_names = X_train.columns
coef = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefficients})
coef['Odds_ratio'] = np.exp(coef['Coefficient'])


coef['Abs_Coef'] = coef['Coefficient'].abs()
top_feat = coef.sort_values(by='Abs_Coef', ascending=False).head(5)

for index, row in top_feat.iterrows():
    print(f"{row['Feature']}: Coefficient = {row['Coefficient']:.4f}, Odds ratio = {row['Odds_ratio']:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))

fig.patch.set_facecolor('#FAF8F5')
ax.set_facecolor('#FAF8F5')

ax.plot(fpr, tpr, color='#FF8C00', lw=2.5, label=f'ROC (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='#FF3366', lw=2, linestyle='--')

ax.grid(color='#E8E2D9', linestyle='-', linewidth=0.5, alpha=0.7)
plt.xlabel("False Positive Rate", fontsize=11, color="#5A5048")
plt.ylabel("True Positive Rate", fontsize=11, color="#5A5048")
plt.title("Churn Prediction", fontsize=14, color='#332C27', pad=20)
plt.legend(loc="upper right",  frameon=False, labelcolor="#5A5048")

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#E8E2D9')
plt.show()