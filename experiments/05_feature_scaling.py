
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from src.optimizers import GradientDescent


# --------------------------------------------------
# 1. Generate synthetic dataset
# --------------------------------------------------

np.random.seed(42)

n = 200

X = np.random.randn(n, 2)

# Give the second feature a much larger scale
X[:, 1] *= 1000

true_theta = np.array([3.0, 0.005])

y = X @ true_theta + np.random.randn(n) * 0.1

# Add bias column
X = np.column_stack([np.ones(n), X])


# --------------------------------------------------
# 2. Feature scaling
# --------------------------------------------------

mean = X[:, 1:].mean(axis=0)
std = X[:, 1:].std(axis=0)

X_scaled = X.copy()

X_scaled[:, 1:] = (X[:, 1:] - mean) / std


# --------------------------------------------------
# 3. Objective and gradient
# --------------------------------------------------

def loss(A, theta):
    errors = A @ theta - y
    return np.mean(errors ** 2) / 2


def gradient(A, theta):
    errors = A @ theta - y
    return A.T @ errors / len(y)


# --------------------------------------------------
# 4. Gradient Descent
# --------------------------------------------------

def run_gd(A, learning_rate, iterations):

    theta = np.zeros(A.shape[1])
    losses = []

    for _ in range(iterations):

        current_loss = loss(A, theta)
        losses.append(current_loss)

        grad = gradient(A, theta)

        theta -= learning_rate * grad

        # Stop if numerical instability occurs
        if not np.all(np.isfinite(theta)):
            break

    return theta, np.array(losses)


# --------------------------------------------------
# 5. Run experiments
# --------------------------------------------------

iterations = 2000

# Smaller learning rate for unscaled data
raw_theta, raw_losses = run_gd(
    X,
    learning_rate=1e-7,
    iterations=iterations
)

# Larger learning rate is possible after scaling
scaled_theta, scaled_losses = run_gd(
    X_scaled,
    learning_rate=0.1,
    iterations=iterations
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results = Path("results")
results.mkdir(exist_ok=True)

# Save loss histories separately because their
# lengths may differ if one run stops early.

pd.DataFrame({
    "iteration": np.arange(len(raw_losses)),
    "loss": raw_losses
}).to_csv(
    results / "05_unscaled_losses.csv",
    index=False
)

pd.DataFrame({
    "iteration": np.arange(len(scaled_losses)),
    "loss": scaled_losses
}).to_csv(
    results / "05_scaled_losses.csv",
    index=False
)


# --------------------------------------------------
# 7. Plot convergence
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.semilogy(
    raw_losses,
    label="Unscaled features"
)

plt.semilogy(
    scaled_losses,
    label="Scaled features"
)

plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Effect of Feature Scaling on Gradient Descent")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    results / "05_feature_scaling.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 8. Print final results
# --------------------------------------------------

print("Unscaled features:")
print("Final loss:", loss(X, raw_theta))
print("Parameters:", raw_theta)

print("\nScaled features:")
print("Final loss:", loss(X_scaled, scaled_theta))
print("Parameters:", scaled_theta)
