# # Import necessary libraries
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_wine
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score

# # Load the Wine dataset
# data = load_wine()
# X = data.data  # Features
# y = data.target  # Target labels

# # Display feature names and target variable information
# print("\n" + "="*60)
# print("Dataset Information")
# print("="*60)
# print("\nFeatures:")
# for name in data.feature_names:
#     print(f"  - {name}")

# print("\nTarget variable: wine class")
# print("Classes:")
# for idx, name in enumerate(data.target_names):
#     print(f"  {idx}: {name}")

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create a neural network classifier
# mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, random_state=42)

# # Train the model
# mlp.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = mlp.predict(X_test)

# # Pretty print the results
# print("\n" + "="*60)
# print("Neural Network Predictions - Result obtained by Trey Lumley")
# print("="*60)
# print(f"\nTotal predictions: {len(y_pred)}")
# print(f"\nPredictions (class labels):")
# print("-" * 60)
# for i, pred in enumerate(y_pred, 1):
#     class_name = data.target_names[pred]
#     print(f"  Sample {i:2d}: Class {pred} ({class_name})")
# print("-" * 60) 


# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

"""
Wine Classification using a Neural Network (MLPClassifier)

This script:
1. Loads the Wine dataset from scikit-learn.
2. Prints dataset feature names and target classes.
3. Splits the data into training and test sets.
4. Trains a neural network classifier (MLP).
5. Evaluates the model using accuracy and a confusion matrix.
"""

# Import necessary libraries
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    # load_wine() returns a "Bunch" object (similar to a dictionary)
    data = load_wine()

    # Features (X): all numeric measurements (shape: [n_samples, n_features])
    X = data.data

    # Target (y): wine class labels (0, 1, 2)
    y = data.target

    # ------------------------------------------------------------------
    # 2. Print dataset information (features and target)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print("\nNumber of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    print("\nFeatures:")
    for name in data.feature_names:
        print(f"  - {name}")

    print("\nTarget variable: wine class")
    print("Classes:")
    for idx, name in enumerate(data.target_names):
        print(f"  {idx}: {name}")

    # ------------------------------------------------------------------
    # 3. Split into training and testing sets
    # ------------------------------------------------------------------
    # We hold out 30% of the data for testing.
    # random_state is set for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y  # keep class proportions similar in train and test
    )

    print("\n" + "=" * 60)
    print("Data Split")
    print("=" * 60)
    print("Training samples:", X_train.shape[0])
    print("Test samples:    ", X_test.shape[0])

    # ------------------------------------------------------------------
    # 4. Define and train the Neural Network classifier
    # ------------------------------------------------------------------
    # MLPClassifier is a feed-forward neural network.
    # - hidden_layer_sizes=(30,) sets one hidden layer with 30 neurons.
    # - activation='relu' is a common choice for hidden layers.
    # - solver='adam' is a good default optimizer.
    # - max_iter controls the maximum number of training iterations.
    # - random_state for reproducibility.
    mlp = MLPClassifier(
        hidden_layer_sizes=(30,),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    print("\n" + "=" * 60)
    print("Training the Neural Network")
    print("=" * 60)

    mlp.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Make predictions on the test set
    # ------------------------------------------------------------------
    y_pred = mlp.predict(X_test)

    # ------------------------------------------------------------------
    # 6. Evaluate the model
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("Neural Network Evaluation - Wine Classification")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix (rows: true class, columns: predicted class):")
    print(cm)

    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        zero_division=0  # tell sklearn to use 0 instead of warning when a class has no predicted samples
    ))

    # ------------------------------------------------------------------
    # 7. (Optional) Show a few predictions with their true labels
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Sample Predictions (first 10 test examples)")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        true_label = data.target_names[y_test[i]]
        pred_label = data.target_names[y_pred[i]]
        print(f"Sample {i+1:2d}: True = {true_label:10s} | Predicted = {pred_label:10s}")

if __name__ == "__main__":
    main()