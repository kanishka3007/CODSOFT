import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix



data = pd.read_csv("creditcard fraud detection.csv")

print("===================================")
print(" Dataset Loaded Successfully")
print("===================================\n")

print(data.head())



print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())



X = data.drop("Class", axis=1)
y = data["Class"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)



predictions = model.predict(X_test)

print("\n===================================")
print(" First 10 Predictions")
print("===================================\n")

for actual, predicted in zip(y_test.values[:10], predictions[:10]):
    print(f"Actual: {actual}   Predicted: {predicted}")

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print(" Model Performance")
print("===================================")

print(f"Accuracy : {accuracy:.4f}")



cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix")
print(cm)


new_transaction = X.iloc[[0]]

prediction = model.predict(new_transaction)

print("\n===================================")
print(" New Prediction")
print("===================================")

if prediction[0] == 0:
    print("Transaction is Genuine")
else:
    print("Transaction is Fraudulent")



class_counts = data["Class"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(["Genuine", "Fraud"], class_counts.values)

plt.title("Credit Card Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()