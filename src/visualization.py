"""Plotting helpers for optimization experiments."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(curves, output_path, title="Loss convergence"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, losses in curves.items():
        values = np.maximum(np.asarray(losses, dtype=float), 1e-16)
        plt.plot(values, label=label)
    plt.xlabel("Iteration / epoch")
    plt.ylabel("Objective value")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_contours_and_paths(objective, paths, output_path, xlim=(-2, 2), ylim=(-2, 2), title="Optimization paths"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    xs, ys = np.linspace(*xlim, 220), np.linspace(*ylim, 220)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.empty_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = objective(np.array([xx[i, j], yy[i, j]]))
    positive = zz[zz > 0]
    levels = np.geomspace(max(1e-8, positive.min()), max(positive.max(), 1e-7), 28)
    plt.figure(figsize=(7, 6))
    plt.contour(xx, yy, zz, levels=levels)
    for name, path in paths.items():
        plt.plot(path[:, 0], path[:, 1], marker=".", markersize=2, label=name)
        plt.scatter(path[0, 0], path[0, 1], marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
