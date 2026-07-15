
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("Movies.csv", encoding="latin1")

print("===================================")
print(" Dataset Loaded Successfully")
print("===================================\n")

print(data.head())



print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())


data = data[["Duration", "Votes", "Rating"]]


data = data.dropna()

data["Duration"] = data["Duration"].str.replace(" min", "", regex=False)


data["Duration"] = pd.to_numeric(data["Duration"], errors="coerce")
data["Votes"] = data["Votes"].str.replace(",", "", regex=False)
data["Votes"] = pd.to_numeric(data["Votes"], errors="coerce")
data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")


data = data.dropna()

print("\nCleaned Dataset")
print(data.head())


X = data[["Duration", "Votes"]]
y = data["Rating"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)



predictions = model.predict(X_test)

print("\n===================================")
print(" First 10 Predictions")
print("===================================\n")

for actual, predicted in zip(y_test.values[:10], predictions[:10]):
    print(f"Actual: {actual:.2f}   Predicted: {predicted:.2f}")



r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\n===================================")
print(" Model Performance")
print("===================================")

print(f"R² Score           : {r2:.4f}")
print(f"Mean Squared Error : {mse:.4f}")



print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature} : {coef:.6f}")


new_movie = pd.DataFrame({
    "Duration": [150],
    "Votes": [50000]
})

prediction = model.predict(new_movie)

print("\n===================================")
print(" New Prediction")
print("===================================")

print(f"Predicted Rating = {prediction[0]:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Rating")

plt.grid(True)

plt.show()
