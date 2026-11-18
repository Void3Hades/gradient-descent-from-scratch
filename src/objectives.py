"""Objective functions used in the optimization experiments."""
import numpy as np


def quadratic(x):
    """f(x) = sum(x_i^2); global minimum at the zero vector."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def quadratic_2d(point):
    """f(x, y) = x^2 + y^2; global minimum at (0, 0)."""
    x, y = np.asarray(point, dtype=float)
    return float(x**2 + y**2)


def rosenbrock(point):
    """Rosenbrock banana function; global minimum at (1, 1)."""
    x, y = np.asarray(point, dtype=float)
    return float((1.0 - x)**2 + 100.0 * (y - x**2)**2)


def mse_loss(X, y, theta):
    """J(theta) = ||X theta-y||^2/(2n)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    residual = X @ theta - y
    return float(np.mean(residual**2) / 2.0)
