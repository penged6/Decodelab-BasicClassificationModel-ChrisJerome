"""
Basic Classification Model
---------------------------
Loads a small dataset (Iris), splits it into training/testing sets,
and trains a simple classifier (K-Nearest Neighbors) to predict flower species.

How to run:
    python classification_model.py
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Load and understand the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("First 5 rows of the dataset:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Classes: {list(iris.target_names)}\n")

X = iris.data          # features (sepal length, sepal width, petal length, petal width)
y = iris.target        # labels (species: 0, 1, 2)

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

# 3. Apply a simple classification algorithm (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2%}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Try a single prediction example
sample = [X_test[0]]
prediction = model.predict(sample)
print(f"Sample input: {sample[0]}")
print(f"Predicted species: {iris.target_names[prediction[0]]}")
print(f"Actual species: {iris.target_names[y_test[0]]}")
