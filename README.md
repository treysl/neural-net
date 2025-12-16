# Neural Network Classification Project

This project demonstrates neural network classification using scikit-learn's MLPClassifier on various datasets. The project includes implementations for wine classification and iris classification tasks.

## Project Structure

```
neural-net/
├── pyscript.py              # Main script with comprehensive wine classification
├── pyscript_task1.py        # Task 1: Basic wine classification implementation
├── pyscript_task2.py        # Task 2: Enhanced wine classification with detailed evaluation
├── results_task1_floral.txt # Results from Task 1 (Iris dataset)
├── results_task1_wine.txt   # Results from Task 1 (Wine dataset)
└── results_task2_classify_wine.txt # Results from Task 2 (Wine dataset)
```

## Overview

This project implements multi-layer perceptron (MLP) neural networks for classification tasks using scikit-learn. The scripts demonstrate:

- Loading datasets from scikit-learn
- Data preprocessing and train-test splitting
- Training neural network classifiers
- Model evaluation with various metrics
- Prediction and result visualization

## Scripts

### `pyscript.py`
Main script with a comprehensive implementation of wine classification. Features include:
- Detailed dataset information display
- Stratified train-test splitting
- MLPClassifier with configurable parameters
- Comprehensive evaluation (accuracy, confusion matrix, classification report)
- Sample prediction display

### `pyscript_task1.py`
Basic implementation for Task 1. This script:
- Loads the Wine dataset
- Displays dataset features and classes
- Trains a simple MLPClassifier
- Makes predictions and calculates accuracy
- Prints formatted prediction results

### `pyscript_task2.py`
Enhanced implementation for Task 2 with improved evaluation metrics. This script:
- Loads the Wine dataset
- Provides detailed dataset information
- Uses stratified splitting for balanced train/test sets
- Trains an MLPClassifier with explicit hyperparameters
- Evaluates using accuracy, confusion matrix, and classification report
- Displays sample predictions with true vs predicted labels

## Datasets

### Wine Dataset
- **Samples**: 178
- **Features**: 13 (alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline)
- **Classes**: 3 wine classes (class_0, class_1, class_2)
- **Source**: scikit-learn's `load_wine()` function

### Iris Dataset (Task 1 Floral)
- **Samples**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 iris species (setosa, versicolor, virginica)
- **Source**: scikit-learn's `load_iris()` function

## Requirements

```python
numpy
scikit-learn
```

Install dependencies using:
```bash
pip install numpy scikit-learn
```

## Usage

### Run Task 1
```bash
python pyscript_task1.py
```

### Run Task 2
```bash
python pyscript_task2.py
```

### Run Main Script
```bash
python pyscript.py
```

## Model Configuration

The neural network models use the following default configuration:

- **Architecture**: Single hidden layer with 30 neurons
- **Activation**: ReLU (Rectified Linear Unit)
- **Solver**: Adam optimizer
- **Max Iterations**: 1000
- **Random State**: 42 (for reproducibility)
- **Train/Test Split**: 70/30

## Results

### Task 1 - Floral (Iris Dataset)
- **Accuracy**: 100%
- Perfect classification achieved on the iris dataset

### Task 1 - Wine Dataset
- **Accuracy**: 33.33%
- Model struggled with class imbalance, predicting only class_0

### Task 2 - Wine Classification
- **Accuracy**: 33.33%
- Includes detailed confusion matrix and classification report
- Shows the need for hyperparameter tuning and feature scaling

## Notes

- The wine classification results show that the model may need:
  - Feature scaling/normalization
  - Hyperparameter tuning
  - Different network architecture
  - More training iterations or different solver

- Results files contain the output from previous runs and can be used for comparison.

## Author

Trey Lumley

## License

This project is for educational purposes.
