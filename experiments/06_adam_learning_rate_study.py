"""Compare Adam learning rates on Rosenbrock."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.objectives import rosenbrock
from src.gradients import rosenbrock_gradient
from src.optimizers import Adam, optimize
from src.visualization import plot_loss_curves

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
results.mkdir(exist_ok=True)
curves, rows = {}, []
for lr in [0.003, 0.01, 0.03, 0.1]:
    params, losses, _ = optimize(rosenbrock, rosenbrock_gradient, np.array([-1.2, 1.0]), Adam(lr), steps=20000, tolerance=1e-8)
    curves[f"Adam lr={lr}"] = losses
    rows.append({"learning_rate": lr, "x": params[0], "y": params[1], "final_loss": losses[-1]})
plot_loss_curves(curves, results / "06_adam_learning_rates.png", "Adam learning-rate comparison")
pd.DataFrame(rows).to_csv(results / "06_adam_metrics.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
