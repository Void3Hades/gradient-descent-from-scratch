"""Study learning rates on f(x)=x^2."""
from pathlib import Path
import numpy as np
from src.objectives import quadratic
from src.gradients import quadratic_gradient
from src.optimizers import GradientDescent, optimize
from src.visualization import plot_loss_curves

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
results.mkdir(exist_ok=True)
curves = {}
for lr in [0.05, 0.1, 0.4, 0.9]:
    _, losses, _ = optimize(quadratic, quadratic_gradient, np.array([5.0]), GradientDescent(lr), steps=100)
    curves[f"learning rate={lr}"] = losses
    print(f"lr={lr}: final loss={losses[-1]:.8g}")
plot_loss_curves(curves, results / "01_quadratic_learning_rates.png", "Gradient descent on f(x)=x²")
