"""Optimization algorithms implemented using NumPy only."""
import numpy as np


class Optimizer:
    def update(self, parameters, gradient):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.lr = float(learning_rate)

    def update(self, parameters, gradient):
        return parameters - self.lr * gradient


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr, self.beta = float(learning_rate), float(beta)
        self.velocity = None

    def update(self, parameters, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
        self.velocity = self.beta * self.velocity + gradient
        return parameters - self.lr * self.velocity


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.lr, self.beta, self.eps = float(learning_rate), float(beta), float(epsilon)
        self.squared_gradient = None

    def update(self, parameters, gradient):
        if self.squared_gradient is None:
            self.squared_gradient = np.zeros_like(parameters)
        self.squared_gradient = self.beta * self.squared_gradient + (1.0 - self.beta) * gradient**2
        return parameters - self.lr * gradient / (np.sqrt(self.squared_gradient) + self.eps)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = float(learning_rate)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(epsilon)
        self.m = self.v = None
        self.t = 0

    def update(self, parameters, gradient):
        if self.m is None:
            self.m, self.v = np.zeros_like(parameters), np.zeros_like(parameters)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return parameters - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def optimize(objective, gradient_fn, initial_parameters, optimizer, steps=1000, tolerance=1e-10):
    """Run an optimizer and return final parameters, objective history and parameter path."""
    params = np.asarray(initial_parameters, dtype=float).copy()
    losses, path = [float(objective(params))], [params.copy()]
    for _ in range(int(steps)):
        grad = np.asarray(gradient_fn(params), dtype=float)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("Non-finite gradient encountered.")
        params = optimizer.update(params, grad)
        loss = float(objective(params))
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite objective; try a smaller learning rate.")
        losses.append(loss)
        path.append(params.copy())
        if np.linalg.norm(grad) < tolerance:
            break
    return params, np.asarray(losses), np.asarray(path)


def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, epochs=100, batch_size=1, seed=42):
    """Mini-batch SGD for linear regression; batch_size=1 gives SGD."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).copy()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    rng = np.random.default_rng(seed)
    losses = [float(np.mean((X @ theta - y)**2) / 2.0)]
    for _ in range(epochs):
        indices = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = indices[start:start + batch_size]
            xb, yb = X[batch], y[batch]
            grad = xb.T @ (xb @ theta - yb) / len(batch)
            theta -= learning_rate * grad
        losses.append(float(np.mean((X @ theta - y)**2) / 2.0))
    return theta, np.asarray(losses)
