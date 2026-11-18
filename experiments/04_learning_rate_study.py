"""Study convergence and divergence as the learning rate changes."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.objectives import quadratic
from src.gradients import quadratic_gradient
from src.optimizers import GradientDescent, optimize
from src.visualization import plot_loss_curves

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
results.mkdir(exist_ok=True)
curves, rows = {}, []
for lr in [0.05, 0.2, 0.5, 0.9, 1.01]:
    try:
        _, losses, _ = optimize(quadratic, quadratic_gradient, np.array([4.0]), GradientDescent(lr), steps=50)
        curves[f"lr={lr}"] = losses
        rows.append({"learning_rate": lr, "final_loss": losses[-1], "finite": bool(np.all(np.isfinite(losses)))})
    except FloatingPointError:
        rows.append({"learning_rate": lr, "final_loss": np.nan, "finite": False})
plot_loss_curves(curves, results / "04_learning_rate_study.png", "Effect of learning rate on f(x)=x²")
pd.DataFrame(rows).to_csv(results / "04_learning_rate_metrics.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
