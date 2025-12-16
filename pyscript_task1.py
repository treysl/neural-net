# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the Wine dataset
data = load_wine()
X = data.data  # Features
y = data.target  # Target labels

# Display feature names and target variable information
print("\n" + "="*60)
print("Dataset Information")
print("="*60)
print("\nFeatures:")
for name in data.feature_names:
    print(f"  - {name}")

print("\nTarget variable: wine class")
print("Classes:")
for idx, name in enumerate(data.target_names):
    print(f"  {idx}: {name}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a neural network classifier
mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Pretty print the results
print("\n" + "="*60)
print("Neural Network Predictions - Result obtained by Trey Lumley")
print("="*60)
print(f"\nTotal predictions: {len(y_pred)}")
print(f"\nPredictions (class labels):")
print("-" * 60)
for i, pred in enumerate(y_pred, 1):
    class_name = data.target_names[pred]
    print(f"  Sample {i:2d}: Class {pred} ({class_name})")
print("-" * 60) 


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
