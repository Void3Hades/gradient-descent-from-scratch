import numpy as np
from src.gradients import quadratic_gradient, quadratic_2d_gradient, rosenbrock_gradient, mse_gradient


def test_quadratic_gradient():
    assert np.isclose(quadratic_gradient(3.0), 6.0)


def test_quadratic_2d_gradient():
    assert np.allclose(quadratic_2d_gradient([2.0, 3.0]), [4.0, 6.0])


def test_rosenbrock_gradient_at_minimum():
    assert np.allclose(rosenbrock_gradient([1.0, 1.0]), [0.0, 0.0])


def test_mse_gradient():
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 3.0])
    theta = np.array([0.0, 0.0])
    assert np.allclose(mse_gradient(X, y, theta), [-2.0, -1.5])
