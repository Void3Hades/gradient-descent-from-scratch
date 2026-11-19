
"""Multi-seed comparison of optimizers on the Rosenbrock function."""

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

# Each optimizer gets a fresh instance for every run.
OPTIMIZERS = {
    "Gradient Descent": lambda: GradientDescent(learning_rate=0.001),
    "Momentum": lambda: Momentum(learning_rate=0.0005, beta=0.9),
    "RMSProp": lambda: RMSProp(learning_rate=0.01),
    "Adam": lambda: Adam(learning_rate=0.03),
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

print(f"Running {N_RUNS} initial points for each optimizer.")
print(f"Maximum steps: {MAX_STEPS}")
print(f"Random seed: {RANDOM_SEED}\n")


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

all_results = []

for run_id, initial_point in enumerate(initial_points, start=1):

    print(f"Initial point {run_id}/{N_RUNS}: {initial_point}")

    for name, optimizer_factory in OPTIMIZERS.items():

        try:
            optimizer = optimizer_factory()

            params, losses, path = optimize(
                objective=rosenbrock,
                gradient_fn=rosenbrock_gradient,
                initial_parameters=initial_point.copy(),
                optimizer=optimizer,
                steps=MAX_STEPS,
                tolerance=TOLERANCE,
            )

            final_loss = float(rosenbrock(params))
            final_gradient_norm = float(
                np.linalg.norm(rosenbrock_gradient(params))
            )

            iterations = len(losses) - 1
            converged = final_gradient_norm < TOLERANCE

            status = "converged" if converged else "max_steps"

        except (FloatingPointError, ValueError, OverflowError) as error:

            params = np.array([np.nan, np.nan])
            final_loss = np.nan
            final_gradient_norm = np.nan
            iterations = 0
            converged = False
            status = f"failed: {type(error).__name__}"

        all_results.append({
            "run_id": run_id,
            "initial_x": initial_point[0],
            "initial_y": initial_point[1],
            "optimizer": name,
            "final_x": params[0],
            "final_y": params[1],
            "final_loss": final_loss,
            "final_gradient_norm": final_gradient_norm,
            "iterations": iterations,
            "converged": converged,
            "status": status,
        })

        print(
            f"  {name}: loss={final_loss:.6g}, "
            f"iterations={iterations}, status={status}"
        )


# --------------------------------------------------
# Save individual run results
# --------------------------------------------------

individual_file = RESULTS / "07_multiseed_individual_results.csv"

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

for name in OPTIMIZERS:

    runs = [
        result for result in all_results
        if result["optimizer"] == name
    ]

    successful = [
        result for result in runs
        if np.isfinite(result["final_loss"])
    ]

    losses = [result["final_loss"] for result in successful]
    iterations = [result["iterations"] for result in successful]

    converged_count = sum(result["converged"] for result in runs)

    if losses:
        mean_loss = float(np.mean(losses))
        median_loss = float(np.median(losses))
        std_loss = float(np.std(losses))
        mean_iterations = float(np.mean(iterations))
    else:
        mean_loss = median_loss = std_loss = mean_iterations = np.nan

    summary.append({
        "optimizer": name,
        "successful_runs": len(successful),
        "converged_runs": converged_count,
        "convergence_rate": converged_count / N_RUNS,
        "mean_final_loss": mean_loss,
        "median_final_loss": median_loss,
        "std_final_loss": std_loss,
        "mean_iterations": mean_iterations,
    })


summary_file = RESULTS / "07_multiseed_summary.csv"

with open(summary_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# Plot 1: Final loss comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

plot_names = []
plot_losses = []

for name in OPTIMIZERS:
    values = [
        result["final_loss"]
        for result in all_results
        if result["optimizer"] == name
        and np.isfinite(result["final_loss"])
    ]

    if values:
        plot_names.append(name)
        plot_losses.append(values)

ax.boxplot(plot_losses, tick_labels=plot_names)
ax.set_yscale("log")
ax.set_ylabel("Final objective value (log scale)")
ax.set_title("Final Loss Across 20 Initial Points")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    RESULTS / "07_multiseed_final_loss.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# Plot 2: Convergence rate comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

names = [item["optimizer"] for item in summary]
rates = [
    item["convergence_rate"] * 100
    for item in summary
]

ax.bar(names, rates)
ax.set_ylabel("Convergence rate (%)")
ax.set_ylim(0, 105)
ax.set_title("Optimizer Convergence Rate Across 20 Runs")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    RESULTS / "07_multiseed_convergence_rate.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("MULTI-SEED EXPERIMENT SUMMARY")
print("=" * 75)

for item in summary:
    print(
        f"{item['optimizer']:18s} | "
        f"Converged: {item['converged_runs']:2d}/{N_RUNS} | "
        f"Mean loss: {item['mean_final_loss']:.6g} | "
        f"Median loss: {item['median_final_loss']:.6g}"
    )

print("\nSaved files:")
print(individual_file)
print(summary_file)
print(RESULTS / "07_multiseed_final_loss.png")
print(RESULTS / "07_multiseed_convergence_rate.png")
