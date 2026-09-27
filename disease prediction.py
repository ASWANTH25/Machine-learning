import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("disease_dataset_3000.csv")

print("Dataset Preview:")
print(df.head())

X = df[["Fever","Cough","BP","Sugar"]]
y = df["Result"]

y = y.map({"No Disease":0, "Disease":1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n--- NAIVE BAYES ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("\n--- KNN ---")
print("Accuracy:", accuracy_score(y_test, knn_pred))

print("\nEnter Patient Details")

fever = int(input("Fever (0/1): "))
cough = int(input("Cough (0/1): "))
bp = int(input("BP: "))
sugar = int(input("Sugar: "))

sample = [[fever, cough, bp, sugar]]

nb_result = nb_model.predict(sample)
knn_result = knn_model.predict(sample)

print("\nNaive Bayes Prediction:",
      "Disease" if nb_result[0]==1 else "No Disease")

print("KNN Prediction:",
      "Disease" if knn_result[0]==1 else "No Disease")
