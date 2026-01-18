# MNIST Digit Recognition

A simple neural network implementation from scratch using NumPy for recognizing handwritten digits from the MNIST dataset.

## Overview

This project implements a 2-layer fully connected neural network to classify handwritten digits (0-9) with:
- **Input layer**: 784 neurons (28x28 pixel images)
- **Hidden layer**: 128 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

## Dataset

The project uses the MNIST dataset in CSV format:
- `mnist_train.csv` - Training data
- `mnist_test.csv` - Test data

## Usage

Run the main script to train and evaluate the model:

```bash
python main.py
```

The script automatically:
- Normalizes input data to [0,1]
- Splits training data (80% train, 20% validation)
- Trains the model with different learning rates (0.1 to 3.0 in steps of 0.2)
- Evaluates accuracy on the test set

## Hyperparameters

- **Learning rate**: 0.1 - 3.0 (grid search)
- **Epochs**: 40
- **Batch size**: 64

## Training

The model uses mini-batch gradient descent with:
- Cross-entropy loss (via softmax)
- Backpropagation for weight updates
- Validation set for monitoring performance

## Results

The final output displays test accuracy for each learning rate configuration.
