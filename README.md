Basic Classification Model

A simple supervised learning project that trains a K-Nearest Neighbors classifier on the Iris flower dataset to predict species from flower measurements.

Requirements:

Python 3.8+
scikit-learn
pandas

Install dependencies:
pip install scikit-learn pandas

Concepts Demonstrated:

Data handling, supervised learning basics, model training

What It Does

1. Load and understand the data — loads the built-in Iris dataset (150 samples, 4 features: sepal length/width, petal length/width, across 3 species) and previews it
2. Split the data — divides it into 80% training and 20% testing sets
3. Train a classifier — fits a K-Nearest Neighbors model (k=3) on the training data
4. Evaluate — predicts on the test set and prints accuracy plus a full classification report (precision, recall, f1-score per class)
5. Demo prediction — runs one sample through the trained model and compares the prediction to the actual label

How to Run:

python classification_model.py

Example Output:

Model accuracy: 100.00%

Classification report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

Sample input: [6.1 2.8 4.7 1.2]
Predicted species: versicolor
Actual species: versicolor

Why KNN?

K-Nearest Neighbors classifies a new data point by looking at the “k” closest points in the training data and taking a majority vote of their labels. It’s one of the simplest classification algorithms to understand, making it a good starting point for learning supervised learning concepts.
