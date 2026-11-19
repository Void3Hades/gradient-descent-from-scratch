
"""Fine-grained learning-rate analysis for four optimizers."""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.objectives import rosenbrock
from src.gradients import rosenbrock_gradient
from src.optimizers import GradientDescent, Momentum, RMSProp, Adam, optimize


# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_RUNS = 20
RANDOM_SEED = 42
MAX_STEPS = 20000
TOLERANCE = 1e-8

CONFIGS = {
    "Gradient Descent": {
        "rates": [0.0008, 0.001, 0.0012, 0.0015],
        "factory": lambda lr: GradientDescent(learning_rate=lr),
    },
    "Momentum": {
        "rates": [0.0005, 0.0008, 0.001, 0.0015, 0.002],
        "factory": lambda lr: Momentum(learning_rate=lr, beta=0.9),
    },
    "RMSProp": {
        "rates": [0.0005, 0.0008, 0.001, 0.0015],
        "factory": lambda lr: RMSProp(learning_rate=lr),
    },
    "Adam": {
        "rates": [0.01, 0.03, 0.05, 0.1],
        "factory": lambda lr: Adam(learning_rate=lr),
    },
}


# --------------------------------------------------
# Generate reproducible initial points
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

initial_points = rng.uniform(
    low=[-1.5, 0.5],
    high=[-0.5, 1.5],
    size=(N_RUNS, 2),
)

all_results = []

print("=" * 75)
print("FINE-GRAINED LEARNING-RATE ANALYSIS")
print("=" * 75)

total_configs = sum(len(c["rates"]) for c in CONFIGS.values())
print(f"Learning-rate configurations: {total_configs}")
print(f"Initial points per configuration: {N_RUNS}")
print(f"Total runs: {total_configs * N_RUNS}")


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

for optimizer_name, config in CONFIGS.items():

    print(f"\nTesting {optimizer_name}")

    for learning_rate in config["rates"]:

        print(f"  Learning rate: {learning_rate}")

        for run_id, initial_point in enumerate(initial_points, start=1):

            try:
                optimizer = config["factory"](learning_rate)

                params, losses, path = optimize(
                    objective=rosenbrock,
                    gradient_fn=rosenbrock_gradient,
                    initial_parameters=initial_point.copy(),
                    optimizer=optimizer,
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

            all_results.append({
                "run_id": run_id,
                "initial_x": initial_point[0],
                "initial_y": initial_point[1],
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "final_x": params[0],
                "final_y": params[1],
                "final_loss": final_loss,
                "final_gradient_norm": gradient_norm,
                "iterations": iterations,
                "converged": converged,
                "status": status,
            })


# --------------------------------------------------
# Save individual results
# --------------------------------------------------

individual_file = RESULTS / "09_fine_grained_individual.csv"

with open(individual_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=all_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# Calculate summary statistics
# --------------------------------------------------

summary = []

for optimizer_name, config in CONFIGS.items():

    for learning_rate in config["rates"]:

        runs = [
            result for result in all_results
            if result["optimizer"] == optimizer_name
            and result["learning_rate"] == learning_rate
        ]

        successful = [
            result for result in runs
            if np.isfinite(result["final_loss"])
        ]

        losses = [r["final_loss"] for r in successful]
        iterations = [r["iterations"] for r in successful]

        converged_count = sum(r["converged"] for r in runs)

        summary.append({
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "successful_runs": len(successful),
            "failed_runs": len(runs) - len(successful),
            "converged_runs": converged_count,
            "convergence_rate": converged_count / N_RUNS,
            "mean_final_loss": (
                float(np.mean(losses)) if losses else np.nan
            ),
            "median_final_loss": (
                float(np.median(losses)) if losses else np.nan
            ),
            "std_final_loss": (
                float(np.std(losses)) if losses else np.nan
            ),
            "mean_iterations": (
                float(np.mean(iterations)) if iterations else np.nan
            ),
        })


summary_file = RESULTS / "09_fine_grained_summary.csv"

with open(summary_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# Plot 1: Convergence rate
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for optimizer_name in CONFIGS:

    rows = [
        item for item in summary
        if item["optimizer"] == optimizer_name
    ]

    rates = [item["learning_rate"] for item in rows]
    convergence = [
        item["convergence_rate"] * 100 for item in rows
    ]

    ax.plot(
        rates,
        convergence,
        marker="o",
        label=optimizer_name,
    )

ax.set_xscale("log")
ax.set_xlabel("Learning rate (log scale)")
ax.set_ylabel("Convergence rate (%)")
ax.set_ylim(-5, 105)
ax.set_title("Fine-Grained Learning-Rate vs Convergence Rate")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "09_fine_grained_convergence.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# Plot 2: Median final loss
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for optimizer_name in CONFIGS:

    rows = [
        item for item in summary
        if item["optimizer"] == optimizer_name
    ]

    rates = [item["learning_rate"] for item in rows]
    losses = [
        item["median_final_loss"] for item in rows
    ]

    ax.plot(
        rates,
        losses,
        marker="o",
        label=optimizer_name,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Learning rate (log scale)")
ax.set_ylabel("Median final loss (log scale)")
ax.set_title("Fine-Grained Learning-Rate vs Final Loss")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "09_fine_grained_final_loss.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 90)
print("FINE-GRAINED EXPERIMENT SUMMARY")
print("=" * 90)

for item in summary:
    print(
        f"{item['optimizer']:18s} | "
        f"LR: {item['learning_rate']:<8g} | "
        f"Converged: {item['converged_runs']:2d}/{N_RUNS} | "
        f"Mean iterations: {item['mean_iterations']:.1f} | "
        f"Median loss: {item['median_final_loss']:.6g}"
    )

print("\nAll result files saved successfully.")
