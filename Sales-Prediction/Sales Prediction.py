import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("sales-prediction.csv")


if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

print("===================================")
print(" Dataset Loaded Successfully")
print("===================================\n")

print(data.head())


print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())


X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]


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

# ==========================================
# Model Parameters
# ==========================================

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature} : {coef:.4f}")


new_data = pd.DataFrame({
    "TV": [200],
    "Radio": [30],
    "Newspaper": [50]
})

prediction = model.predict(new_data)

print("\n===================================")
print(" New Prediction")
print("===================================")

print(f"Predicted Sales = {prediction[0]:.2f}")


plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")


plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.grid(True)

plt.show()