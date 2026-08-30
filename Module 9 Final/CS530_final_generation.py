import numpy as np

# ============================================================
# Function: neuron(x, weights, bias, activation="sigmoid")
# Purpose: Compute the output of a single neuron given an input
#          vector, weights, and a bias term. Supports two
#          activation functions:
#            - "sigmoid": smooth, differentiable, outputs a
#              probability-like value in (0, 1). Ideal for
#              binary classification and gradient-based learning.
#            - "step": classic hard-threshold activation
#              (Heaviside step function), outputs exactly 0 or 1.
#              Useful for a simple perceptron-style binary
#              classifier and for illustrating decision boundaries.
# Params:
#   x          - input feature vector (1D array-like)
#   weights    - weight vector matching x's dimensionality
#   bias       - scalar bias term
#   activation - "sigmoid" (default) or "step"
# ============================================================
def neuron(x, weights, bias, activation="sigmoid"):
    z = np.dot(x, weights) + bias

    if activation == "sigmoid":
        z = np.clip(z, -500, 500)  # overflow protection
        return 1 / (1 + np.exp(-z))
    elif activation == "step":
        return 1.0 if z >= 0 else 0.0
    else:
        raise ValueError("activation must be 'sigmoid' or 'step'")


# ============================================================
# Function: binary_classifier(X, weights, bias, activation="sigmoid")
# Purpose: Classify each sample in a dataset as class 0 or class 1
#          using the neuron() function defined above. For "sigmoid",
#          predictions are thresholded at 0.5 to produce a binary
#          label. For "step", the neuron already returns a binary
#          output directly.
# Params:
#   X          - dataset of samples (2D array-like), each row a sample
#   weights    - weight vector used by the neuron
#   bias       - scalar bias term used by the neuron
#   activation - "sigmoid" (default) or "step"
# Returns:
#   predictions - 1D numpy array of predicted class labels (0 or 1)
# ============================================================
def binary_classifier(X, weights, bias, activation="sigmoid"):
    X = np.asarray(X)
    predictions = np.zeros(X.shape[0])

    for i, x in enumerate(X):
        output = neuron(x, weights, bias, activation=activation)
        if activation == "sigmoid":
            predictions[i] = 1 if output >= 0.5 else 0
        else:  # step already returns 0 or 1
            predictions[i] = output

    return predictions


# ============================================================
# Function: generate_synthetic_dataset(n_samples=100, n_features=2, seed=42)
# Purpose: Generate a synthetic, linearly separable binary
#          classification dataset for testing/training the
#          neuron and binary_classifier functions.
# Params:
#   n_samples  - number of data points to generate (default 100)
#   n_features - number of input features per sample (default 2)
#   seed       - random seed for reproducibility (default 42)
# Returns:
#   X - 2D numpy array of shape (n_samples, n_features)
#   y - 1D numpy array of binary labels (0 or 1)
# ============================================================
def generate_synthetic_dataset(n_samples=100, n_features=2, seed=42):
    rng = np.random.RandomState(seed)

    # True underlying weights/bias used to generate labels
    true_weights = rng.uniform(-1, 1, size=n_features)
    true_bias = rng.uniform(-1, 1)

    X = rng.randn(n_samples, n_features)
    linear_combo = X.dot(true_weights) + true_bias

    # Add a bit of noise to avoid a perfectly separable trivial dataset
    noise = rng.normal(0, 0.5, size=n_samples)
    y = (linear_combo + noise >= 0).astype(float)

    return X, y


# ============================================================
# Function: calculate_weights(X, y, activation="sigmoid",
#                              learning_rate=0.1, epochs=100)
# Purpose: Learn a weight vector and bias for the binary
#          classifier using simple gradient-based (sigmoid) or
#          perceptron-style (step) weight update rules, based on
#          a synthetic dataset produced by generate_synthetic_dataset().
# Params:
#   X             - dataset of samples (2D array-like), from Function 3
#   y             - true binary labels corresponding to X
#   activation    - "sigmoid" (default) or "step", determines the
#                   update rule used during training
#   learning_rate - step size for weight updates (default 0.1)
#   epochs        - number of passes over the dataset (default 100)
# Returns:
#   weights - learned weight vector (1D numpy array)
#   bias    - learned scalar bias term
# ============================================================
def calculate_weights(X, y, activation="sigmoid", learning_rate=0.1, epochs=100):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            output = neuron(x_i, weights, bias, activation=activation)
            error = y_i - output

            weights += learning_rate * error * x_i
            bias += learning_rate * error

    return weights, bias


# ============================================================
# Example usage
# Step 1: Generate a synthetic dataset
# Step 2: Run calculate_weights() once per activation function
# Step 3: Run binary_classifier() once per associated weight set
# Step 4: Print activation function, learned weights, learned
#         bias, and training accuracy for both activation functions
# ============================================================
if __name__ == "__main__":
    # Step 1: Generate a synthetic dataset
    X, y = generate_synthetic_dataset(n_samples=200, n_features=2)

    # Step 2: Run the weight calculation function twice, once per activation
    sigmoid_weights, sigmoid_bias = calculate_weights(
        X, y, activation="sigmoid", learning_rate=0.1, epochs=200
    )
    step_weights, step_bias = calculate_weights(
        X, y, activation="step", learning_rate=0.1, epochs=200
    )

    # Step 3: Run the binary classifier twice, once per weight set
    sigmoid_predictions = binary_classifier(
        X, sigmoid_weights, sigmoid_bias, activation="sigmoid"
    )
    step_predictions = binary_classifier(
        X, step_weights, step_bias, activation="step"
    )

    # Step 4: Print results for both activation functions
    sigmoid_accuracy = np.mean(sigmoid_predictions == y)
    step_accuracy = np.mean(step_predictions == y)

    print("Activation function: sigmoid")
    print("Learned weights:", sigmoid_weights)
    print("Learned bias:", sigmoid_bias)
    print("Training accuracy:", sigmoid_accuracy)
    print()
    print("Activation function: step")
    print("Learned weights:", step_weights)
    print("Learned bias:", step_bias)
    print("Training accuracy:", step_accuracy)