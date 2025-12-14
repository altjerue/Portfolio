"""
Neural Network from Scratch
============================================================

Implementing a 2-layer neural network to solve the XOR problem.
XOR is NOT linearly separable - a single layer can't solve it.
But a 2-layer network with nonlinearity can!

Network Architecture:
    Input (2) -> Hidden (4) -> Output (1)

    x (2,) -> W1 (4,2) -> z1 (4,) -> tanh -> h1 (4,) -> W2 (1,4) -> z2 (1,) -> y_pred
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# %% DATA GENERATION - XOR Problem
# ============================================================
# XOR truth table:
# Input: (0,0) -> Output: 0
# Input: (0,1) -> Output: 1
# Input: (1,0) -> Output: 1
# Input: (1,1) -> Output: 0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

print("XOR Problem:")
print("Inputs:\n", X)
print("Targets:\n", y)
print("\nThis is NOT linearly separable - you can't draw a line to separate them!")
print("-" * 60)


# ============================================================
# %% NETWORK PARAMETERS
# ============================================================
input_size = 2  # 2 inputs (x1, x2)
hidden_size = 4  # 4 hidden neurons
output_size = 1  # 1 output (binary classification)
learning_rate = 0.1
epochs = 10000


# ============================================================
# %% WEIGHT INITIALIZATION
# ============================================================
# Shapes: W1 should be (hidden_size, input_size)
#         b1 should be (hidden_size, 1)
#         W2 should be (output_size, hidden_size)
#         b2 should be (output_size, 1)

W1 = 0.5 * np.random.randn(hidden_size, input_size)
b1 = 0.5 * np.random.randn(hidden_size, 1)
W2 = 0.5 * np.random.randn(output_size, hidden_size)
b2 = 0.5 * np.random.randn(output_size, 1)

print(f"\nInitialized weights:")
print(f"W1 shape: {W1.shape if W1 is not None else 'Not implemented'}")
print(f"b1 shape: {b1.shape if b1 is not None else 'Not implemented'}")
print(f"W2 shape: {W2.shape if W2 is not None else 'Not implemented'}")
print(f"b2 shape: {b2.shape if b2 is not None else 'Not implemented'}")
print("-" * 60)


# ============================================================
# %% ACTIVATION FUNCTIONS
# ============================================================
def tanh(x):
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def tanh_derivative(h):
    """
    Derivative of tanh.
    Input: h = tanh(z) (the OUTPUT of tanh, not the input!)
    Output: tanh'(z) = 1 - h^2
    """
    return 1 - h**2


# ============================================================
# %% FORWARD PASS
# ============================================================
def forward(x):
    """
    Forward propagation through the network.

    Args:
        x: input vector (2,) or (2, 1)

    Returns:
        y_pred: prediction (1,)
        cache: dict containing intermediate values for backpropagation
    """
    # Ensure x is column vector (2, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Layer 1 - Linear transformation
    z1 = W1 @ x + b1

    # Layer 1 - Activation
    h1 = tanh(z1)

    # Layer 2 - Linear transformation
    z2 = W2 @ h1 + b2

    y_pred = z2

    # Cache intermediate values for backpropagation
    cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "y_pred": y_pred}

    return y_pred, cache


# ============================================================
# %% LOSS FUNCTION
# ============================================================
def compute_loss(y_true, y_pred):
    """
    Mean Squared Error loss.
    L = (1/2) * (y_true - y_pred)^2
    """
    return 0.5 * np.mean((y_true - y_pred) ** 2)


# ============================================================
# %% BACKWARD PASS (BACKPROPAGATION)
# ============================================================
def backward(y_true, cache):
    """
    Backpropagation to compute gradients.

    Args:
        y_true: true label (1,)
        cache: dictionary from forward pass

    Returns:
        grads: dictionary of gradients for each parameter
    """
    x = cache["x"]
    h1 = cache["h1"]
    y_pred = cache["y_pred"]

    # Output layer gradients
    dLdz2 = y_pred - y_true

    # Gradient for W2
    dLdW2 = dLdz2 @ h1.T

    # Gradient for b2
    dLdb2 = dLdz2

    # Backpropagate to hidden layer
    dLdh1 = W2.T @ dLdz2

    # Apply activation derivative
    dLdz1 = dLdh1 * tanh_derivative(h1)

    # Gradient for W1
    dLdW1 = dLdz1 @ x.T

    # Gradient for b1
    dLdb1 = dLdz1

    grads = {"dLdW1": dLdW1, "dLdb1": dLdb1, "dLdW2": dLdW2, "dLdb2": dLdb2}

    return grads


# ============================================================
# %% GRADIENT DESCENT UPDATE
# ============================================================
def update_weights(grads, learning_rate):
    """
    Update weights using gradient descent.
    W_new = W_old - learning_rate * gradient
    """
    global W1, b1, W2, b2

    W1 = W1 - learning_rate * grads["dLdW1"]
    b1 = b1 - learning_rate * grads["dLdb1"]
    W2 = W2 - learning_rate * grads["dLdW2"]
    b2 = b2 - learning_rate * grads["dLdb2"]


# ============================================================
# %% TRAINING LOOP
# ============================================================
print("\nStarting training...")

losses = []

for epoch in range(epochs):
    epoch_loss = 0

    # Train on each sample
    for i in range(len(X)):
        x_sample = X[i]
        y_true = y[i]

        # 1. Call forward pass
        y_pred, cache = forward(x_sample)

        # 2. Compute loss
        loss = compute_loss(y_true, y_pred)
        epoch_loss += loss

        # 3. Call backward pass
        grads = backward(y_true, cache)

        # 4. Update weights
        update_weights(grads, learning_rate)

    # Average loss over all samples
    avg_loss = epoch_loss / len(X)
    losses.append(avg_loss)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

print("\nTraining complete!")
print("-" * 60)


# ============================================================
# TESTING
# ============================================================
print("\nTesting on XOR inputs:")
for i in range(len(X)):
    x_test = X[i]
    y_true = y[i]
    y_pred, _ = forward(x_test)
    print(f"Input: {x_test}, True: {y_true[0]}, Predicted: {y_pred[0][0]:.4f}")


# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 4))

# Plot 1: Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)

# Plot 2: Decision boundary
plt.subplot(1, 2, 2)
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict for each point in the mesh
Z = np.array(
    [forward(np.array([x, y]))[0][0][0] for x, y in zip(xx.ravel(), yy.ravel())]
)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap="RdYlBu")
plt.scatter(
    X[:, 0], X[:, 1], c=y.ravel(), s=200, edgecolors="black", linewidth=2, cmap="RdYlBu"
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary (Red=0, Blue=1)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()
