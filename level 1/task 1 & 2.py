import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = pd.read_csv("/Users/ikhaisoshuare/Downloads/Data Set For Task/1) iris.csv")

print("First five rows")
print(iris.head())
print("Dataset info")
iris.info()
print("Missing values")
print(iris.isnull().sum())


#TASK 1: PREPROCESSING
translator = LabelEncoder()

iris["species"] = translator.fit_transform(iris["species"])
print("Classes mapped to:", translator.classes_)

X = iris[["petal_length", "petal_width"]]
#X = iris.drop("species", axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
print("Training data: ", len(X_train))
print("Test data: ", len(X_test))

standard = StandardScaler()
X_train_scaled = standard.fit_transform(X_train)
X_test_scaled = standard.transform(X_test)

print("Original first row of trained data: \n", X_train.iloc[0].values)
print("Scaled first row of trained data: \n", X_train_scaled[0])


#TASK 3: KNN MODEL
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_scaled, y_train)
y_predict = knn.predict(X_test_scaled)

print("Results: ")
print("Accuracy score:", accuracy_score(y_test, y_predict))
print("finished!")

print("\nConfusion matrix:\n", confusion_matrix(y_test, y_predict))
print("\nClassification report:\n", classification_report(y_test, y_predict))
