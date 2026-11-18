"""Compare optimizers on the Rosenbrock function."""
from pathlib import Path
import numpy as np
from src.objectives import rosenbrock
from src.gradients import rosenbrock_gradient
from src.optimizers import GradientDescent, Momentum, RMSProp, Adam, optimize
from src.visualization import plot_loss_curves, plot_contours_and_paths

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
results.mkdir(exist_ok=True)
configs = {
    "Gradient Descent": GradientDescent(learning_rate=0.001),
    "Momentum": Momentum(learning_rate=0.0005, beta=0.9),
    "RMSProp": RMSProp(learning_rate=0.01),
    "Adam": Adam(learning_rate=0.03),
}
curves, paths = {}, {}
for name, optimizer in configs.items():
    params, losses, path = optimize(rosenbrock, rosenbrock_gradient, np.array([-1.2, 1.0]), optimizer, steps=20000, tolerance=1e-8)
    curves[name], paths[name] = losses, path
    print(f"{name}: parameters={params}, final loss={losses[-1]:.8g}")
plot_loss_curves(curves, results / "02_rosenbrock_loss.png", "Optimizer comparison on Rosenbrock")
plot_contours_and_paths(rosenbrock, paths, results / "02_rosenbrock_paths.png", xlim=(-1.5, 1.5), ylim=(-0.5, 2.0), title="Optimizer paths on Rosenbrock")
