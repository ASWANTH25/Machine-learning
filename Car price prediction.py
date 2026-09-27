
import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("car_price_1000_dataset.csv")

print("Dataset Preview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nAverage Mileage:", np.mean(df["Mileage"]))
print("Maximum Price:", np.max(df["Price"]))
print("Minimum Price:", np.min(df["Price"]))
sqrt_value = math.sqrt(df["Mileage"][0])
print("\nSquare root of first mileage value:", sqrt_value)

z_scores = stats.zscore(df["Price"])
print("\nFirst 5 Price Z-Scores:")
print(z_scores[:5])

X = df[["Age", "Mileage", "EngineSize", "Horsepower"]]
y = df["Price"]
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("\nEnter Car Details")
age = int(input("Car Age (years): "))
mileage = int(input("Mileage (km): "))
engine = float(input("Engine Size (liters): "))
hp = int(input("Horsepower: "))
prediction = model.predict([[age, mileage, engine, hp]])

print("\nEstimated Car Price:", prediction[0])

