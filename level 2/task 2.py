import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD & SCALE DATA ---
file_path = "/Users/ikhaisoshuare/Downloads/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv"
churn_train = pd.read_csv(file_path)

features = ['Total day minutes', 'Customer service calls']
X = churn_train[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=7, n_init=10)

cluster_labels = kmeans.fit_predict(X_scaled)

churn_train['Cluster'] = cluster_labels

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Total day minutes', 
    y='Customer service calls', 
    hue='Cluster', 
    palette='plasma', 
    data=churn_train, 
    s=100, 
    alpha=0.7 
)

plt.title('Customer Segmentation Profiles (K=3)')
plt.xlabel('Total Day Minutes')
plt.ylabel('Customer Service Calls')
plt.legend(title='Customer Profile')
plt.grid(True)
plt.savefig('clusters.png', dpi=300, bbox_inches='tight')

plt.show()