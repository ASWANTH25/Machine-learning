import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("creditcard_fraud_3000_dataset.csv")

print("Dataset Preview:")
print(df.head())

le_loc = LabelEncoder()
le_dev = LabelEncoder()
le_class = LabelEncoder()

df["Location"] = le_loc.fit_transform(df["Location"])
df["Device"] = le_dev.fit_transform(df["Device"])
df["Class"] = le_class.fit_transform(df["Class"])

X = df[["Amount","Time","Location","Device"]]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n--- NAIVE BAYES ---")
print("Precision:", precision_score(y_test, nb_pred))
print("Recall:", recall_score(y_test, nb_pred))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("\n--- KNN ---")
print("Precision:", precision_score(y_test, knn_pred))
print("Recall:", recall_score(y_test, knn_pred))

print("\nEnter Transaction Details")

amount = float(input("Amount: "))
time = int(input("Time (0-23): "))
loc = int(input("Location (0=Local,1=National,2=International): "))
dev = int(input("Device (0=Mobile,1=Desktop,2=ATM): "))

sample = [[amount, time, loc, dev]]

nb_result = nb_model.predict(sample)
knn_result = knn_model.predict(sample)

print("\nNaive Bayes Prediction:",
      "Fraud" if nb_result[0]==1 else "Normal")

print("KNN Prediction:",
      "Fraud" if knn_result[0]==1 else "Normal")
