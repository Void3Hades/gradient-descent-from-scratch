"""Compare batch GD, SGD and mini-batch SGD on synthetic regression."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import make_regression_data, train_test_split
from src.objectives import mse_loss
from src.gradients import mse_gradient
from src.optimizers import GradientDescent, optimize, stochastic_gradient_descent
from src.visualization import plot_loss_curves

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
results.mkdir(exist_ok=True)
X, y = make_regression_data(n_samples=500, noise=0.5, seed=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, seed=7)
initial = np.zeros(Xtr.shape[1])
_, batch_losses, _ = optimize(lambda t: mse_loss(Xtr, ytr, t), lambda t: mse_gradient(Xtr, ytr, t), initial, GradientDescent(0.05), steps=300)
theta_sgd, sgd_losses = stochastic_gradient_descent(Xtr, ytr, initial, learning_rate=0.01, epochs=100, batch_size=1, seed=7)
theta_mini, mini_losses = stochastic_gradient_descent(Xtr, ytr, initial, learning_rate=0.03, epochs=100, batch_size=32, seed=7)
curves = {"Batch GD (iterations)": batch_losses, "SGD (epochs)": sgd_losses, "Mini-batch SGD (epochs)": mini_losses}
plot_loss_curves(curves, results / "03_batch_sgd_comparison.png", "Linear regression optimization loss")
rows = []
for name, theta in [("SGD", theta_sgd), ("Mini-batch SGD", theta_mini)]:
    rows.append({"algorithm": name, "intercept": theta[0], "slope": theta[1], "test_loss": mse_loss(Xte, yte, theta)})
pd.DataFrame(rows).to_csv(results / "03_regression_test_metrics.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("True generating parameters: intercept=3.0, slope=2.5")
