
"""Investigate RMSProp learning-rate sensitivity on Rosenbrock."""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.objectives import rosenbrock
from src.gradients import rosenbrock_gradient
from src.optimizers import RMSProp, optimize


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_RUNS = 20
RANDOM_SEED = 42
MAX_STEPS = 20000
TOLERANCE = 1e-8

BETA = 0.9

LEARNING_RATES = [
    0.00001,
    0.00005,
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0007,
    0.001,
]

rng = np.random.default_rng(RANDOM_SEED)

initial_points = rng.uniform(
    low=[-1.5, 0.5],
    high=[-0.5, 1.5],
    size=(N_RUNS, 2),
)

results = []

for learning_rate in LEARNING_RATES:

    print(f"\nTesting RMSProp learning_rate={learning_rate}")

    for run_id, initial_point in enumerate(initial_points, start=1):

        try:
            optimizer = RMSProp(
                learning_rate=learning_rate,
                beta=BETA,
            )

            params, losses, path = optimize(
                rosenbrock,
                rosenbrock_gradient,
                initial_point.copy(),
                optimizer,
                steps=MAX_STEPS,
                tolerance=TOLERANCE,
            )

            final_loss = float(rosenbrock(params))
            gradient_norm = float(
                np.linalg.norm(rosenbrock_gradient(params))
            )

            iterations = len(losses) - 1
            converged = gradient_norm < TOLERANCE
            status = "converged" if converged else "max_steps"

        except (FloatingPointError, ValueError, OverflowError) as error:

            params = np.array([np.nan, np.nan])
            final_loss = np.nan
            gradient_norm = np.nan
            iterations = 0
            converged = False
            status = f"failed: {type(error).__name__}"

        results.append({
            "run_id": run_id,
            "beta": BETA,
            "learning_rate": learning_rate,
            "final_x": params[0],
            "final_y": params[1],
            "final_loss": final_loss,
            "gradient_norm": gradient_norm,
            "iterations": iterations,
            "converged": converged,
            "status": status,
        })

        print(
            f"  Run {run_id:02d}: "
            f"loss={final_loss:.6g}, "
            f"iterations={iterations}, "
            f"{status}"
        )




# Save individual results.
with open(
    RESULTS / "11_rmsprop_lr_individual.csv",
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# Calculate summary.
summary = []
for learning_rate in LEARNING_RATES:

    runs = [
        r for r in results
        if r["learning_rate"] == learning_rate
    ]

    successful = [
        r for r in runs if np.isfinite(r["final_loss"])
    ]

    losses = [r["final_loss"] for r in successful]
    iterations = [r["iterations"] for r in successful]

    converged_count = sum(r["converged"] for r in runs)

    summary.append({
        "learning_rate": learning_rate,
        "successful_runs": len(successful),
        "failed_runs": len(runs) - len(successful),
        "converged_runs": converged_count,
        "convergence_rate": converged_count / N_RUNS,
        "mean_final_loss": np.mean(losses) if losses else np.nan,
        "median_final_loss": np.median(losses) if losses else np.nan,
        "mean_iterations": np.mean(iterations) if iterations else np.nan,
    })
with open(
    RESULTS / "11_rmsprop_lr_summary.csv",
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)


# Plot convergence rates.
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    LEARNING_RATES,
    [r["convergence_rate"] * 100 for r in summary],
    marker="o",
)

ax.set_xlabel("RMSProp learning rate")
ax.set_ylabel("Convergence rate (%)")
ax.set_ylim(-5, 105)
ax.set_title("RMSProp: Learning Rate vs Convergence")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    RESULTS / "11_rmsprop_lr_convergence.png",
    dpi=300,
)
plt.close()


print("\nExperiment 11: RMSProp learning-rate analysis complete.")
for row in summary:
    print(row)
