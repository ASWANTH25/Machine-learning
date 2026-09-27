import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("delivery_time_3000_dataset.csv")

print("Dataset Preview:")
print(df.head())

df["Traffic"] = df["Traffic"].map({"Low":1, "Medium":2, "High":3})
df["Weather"] = df["Weather"].map({"Clear":1, "Rainy":2, "Storm":3})

print("\nAverage Distance:", np.mean(df["Distance"]))
print("Square root of first time:", math.sqrt(df["DeliveryTime"][0]))

z = stats.zscore(df["DeliveryTime"])
print("First 5 Z-scores:", z[:5])

X_simple = df[["Distance"]]
y = df["DeliveryTime"]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

print("\n--- SIMPLE REGRESSION ---")
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R2:", r2_score(y_test, y_pred_simple))

X_multi = df[["Distance","Traffic","Weather"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("\n--- MULTIPLE REGRESSION ---")
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R2:", r2_score(y_test, y_pred_multi))

print("\nEnter Delivery Details")

d = int(input("Distance (km): "))
t = int(input("Traffic (1=Low,2=Medium,3=High): "))
w = int(input("Weather (1=Clear,2=Rainy,3=Storm): "))

simple_pred = model_simple.predict([[d]])
multi_pred = model_multi.predict([[d, t, w]])

print("\nPredicted Time (Simple):", simple_pred[0])
print("Predicted Time (Multiple):", multi_pred[0])
