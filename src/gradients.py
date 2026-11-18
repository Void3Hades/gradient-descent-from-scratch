"""Analytical gradients for the objective functions."""
import numpy as np


def quadratic_gradient(x):
    """Gradient of sum(x_i^2): 2x."""
    return 2.0 * np.asarray(x, dtype=float)


def quadratic_2d_gradient(point):
    x, y = np.asarray(point, dtype=float)
    return np.array([2.0 * x, 2.0 * y])


def rosenbrock_gradient(point):
    """Analytical gradient of the two-dimensional Rosenbrock function."""
    x, y = np.asarray(point, dtype=float)
    dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dy = 200.0 * (y - x**2)
    return np.array([dx, dy])


def mse_gradient(X, y, theta):
    """Gradient of ||X theta-y||^2/(2n)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    return X.T @ (X @ theta - y) / X.shape[0]
