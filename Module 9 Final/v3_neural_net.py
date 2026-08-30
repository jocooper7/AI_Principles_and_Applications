# ============================================================
# Imports
# ============================================================
import numpy as np
from scipy import optimize

# ============================================================
# Data and Architecture Setup
# ============================================================
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])

L1 = 3
L2 = 4
L3 = 1

w1 = np.random.randn(L1 * L2)
w2 = np.random.randn(L2 * L3)
b1 = np.zeros(L2)
b2 = np.zeros(L3)

params = np.concatenate([w1, b1, w2, b2])


# ============================================================
# Function: a(x)
# Purpose: Sigmoid activation function
# Note: Input is clipped to a safe range before exponentiation
#       to prevent overflow for large negative x.
# ============================================================
def a(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))


# ============================================================
# Function: f(params, X, L1, L2, L3)
# Purpose: Forward pass through a 2-layer neural network
# Note: Layer sizes are now passed explicitly as parameters
#       instead of relying on global variables.
# ============================================================
def f(params, X, L1, L2, L3):
    w1 = params[:L1*L2].reshape(L1, L2)
    b1 = params[L1*L2:L1*L2+L2]
    w2 = params[L1*L2+L2:L1*L2+L2+L2*L3].reshape(L2, L3)
    b2 = params[-L3:]

    h = a(X.dot(w1) + b1)
    out = a(h.dot(w2) + b2)
    return out


# ============================================================
# Function: loss(params, X, y, L1, L2, L3)
# Purpose: Mean squared error loss between predictions and targets
# Note: Updated to pass L1, L2, L3 through to f().
# ============================================================
def loss(params, X, y, L1, L2, L3):
    pred = f(params, X, L1, L2, L3)
    return np.mean((pred - y)**2)


# ============================================================
# Function: grad(params, X, y, L1, L2, L3)
# Purpose: Numerical gradient of loss() via central differences
# Note: Inefficient — computes 2 full forward+loss passes per
#       parameter (O(n) loss evaluations). Logic left unchanged
#       to preserve existing behavior. Also updated to pass
#       L1, L2, L3 through to loss(). Possible improvements:
#   1. Vectorized/analytical backpropagation
#   2. Vectorized finite differences using perturbation matrices
#   3. Use automatic differentiation (autograd/JAX/TensorFlow)
# ============================================================
def grad(params, X, y, L1, L2, L3):
    eps = 1e-7
    g = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        g[i] = (loss(params_plus, X, y, L1, L2, L3) - loss(params_minus, X, y, L1, L2, L3)) / (2*eps)
    return g


# ============================================================
# Optimization and Evaluation
# ============================================================
result = optimize.minimize(
    loss, params, args=(X, y, L1, L2, L3),
    method='L-BFGS-B', jac=grad,
    options={'maxiter': 1000, 'disp': True}
)

opt_params = result.x

test = np.array([[0,0,1]])
pred = f(opt_params, test, L1, L2, L3)
print("Test:", pred)

print("\nAll:")
print(f(opt_params, X, L1, L2, L3))