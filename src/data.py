"""Synthetic data generation and splitting utilities."""
import numpy as np


def make_regression_data(n_samples=200, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n_samples)
    y = 3.0 + 2.5 * x + rng.normal(0.0, noise, size=n_samples)
    X = np.column_stack([np.ones(n_samples), x])
    return X, y


def train_test_split(X, y, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    n_test = max(1, int(round(len(y) * test_fraction)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
