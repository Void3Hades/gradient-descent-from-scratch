import numpy as np
from src.optimizers import GradientDescent, Momentum, RMSProp, Adam, optimize
from src.objectives import quadratic
from src.gradients import quadratic_gradient


def test_gradient_descent_update():
    updated = GradientDescent(0.1).update(np.array([2.0]), np.array([4.0]))
    assert np.allclose(updated, [1.6])


def test_adaptive_optimizers_finite():
    for optimizer in [Momentum(), RMSProp(), Adam()]:
        value = optimizer.update(np.array([1.0, -1.0]), np.array([0.5, -0.5]))
        assert np.all(np.isfinite(value))


def test_optimize_reduces_loss():
    params, losses, path = optimize(quadratic, quadratic_gradient, np.array([4.0]), GradientDescent(0.1), steps=100)
    assert losses[-1] < losses[0]
    assert abs(params[0]) < 1e-3
    assert len(path) == len(losses)
