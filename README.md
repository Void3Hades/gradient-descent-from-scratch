# Gradient Descent and Optimization Algorithms from Scratch

A research-oriented educational project implementing and experimentally studying optimization algorithms using **Python and NumPy**, without relying on machine-learning optimizer libraries.

The project explores how learning rates, optimizer hyperparameters, initialization, and feature scaling affect optimization performance.

## Algorithms

- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD
- Momentum
- RMSProp
- Adam

## Objective Functions and Datasets

- **Quadratic function:** \(f(x)=\sum_i x_i^2\)
- **Two-dimensional quadratic:** \(f(x,y)=x^2+y^2\)
- **Rosenbrock function:** \(f(x,y)=(1-x)^2+100(y-x^2)^2\)
- **Linear regression:** Mean squared error (MSE)

## Project Structure

```text
gradient-descent-from-scratch/
├── src/
│   ├── objectives.py
│   ├── gradients.py
│   ├── optimizers.py
│   ├── data.py
│   └── visualization.py
├── experiments/
│   ├── 01_quadratic_gradient_descent.py
│   ├── 02_compare_optimizers_rosenbrock.py
│   ├── 03_linear_regression_batch_vs_sgd.py
│   ├── 04_learning_rate_study.py
│   ├── 05_feature_scaling.py
│   ├── 06_adam_learning_rate_study.py
│   ├── 07_multiseed_optimizer_comparison.py
│   ├── 08_learning_rate_sensitivity.py
│   ├── 09_fine_grained_learning_rate.py
│   ├── 10_rmsprop_beta_analysis.py
│   └── 11_rmsprop_learning_rate_analysis.py
├── tests/
├── results/
├── report/
├── requirements.txt
└── README.md
```

## Installation

Python 3.9 or later is recommended.

Clone the repository:

```bash
git clone https://github.com/Void3Hades/gradient-descent-from-scratch.git
cd gradient-descent-from-scratch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

The project currently contains eleven experiments.

| No. | Experiment | Main objective |
|---|---|---|
| 01 | Quadratic Gradient Descent | Study basic gradient descent and learning rates |
| 02 | Optimizer Comparison on Rosenbrock | Compare GD, Momentum, RMSProp, and Adam |
| 03 | Batch GD vs SGD | Study optimization for linear regression |
| 04 | Learning-Rate Study | Investigate the effect of learning rate |
| 05 | Feature Scaling | Examine how scaling affects optimization |
| 06 | Adam Learning-Rate Study | Investigate Adam's learning-rate sensitivity |
| 07 | Multi-Seed Optimizer Comparison | Compare optimizer performance across initial points |
| 08 | Learning-Rate Sensitivity | Study learning-rate sensitivity across optimizers |
| 09 | Fine-Grained Learning-Rate Analysis | Examine narrower learning-rate ranges |
| 10 | RMSProp Beta Analysis | Investigate the effect of RMSProp's decay rate |
| 11 | RMSProp Learning-Rate Analysis | Investigate RMSProp's learning-rate sensitivity |

Run individual experiments from the project root:

```bash
python -m experiments.01_quadratic_gradient_descent
python -m experiments.02_compare_optimizers_rosenbrock
python -m experiments.03_linear_regression_batch_vs_sgd
python -m experiments.04_learning_rate_study
python -m experiments.05_feature_scaling
python -m experiments.06_adam_learning_rate_study
python -m experiments.07_multiseed_optimizer_comparison
python -m experiments.08_learning_rate_sensitivity
python -m experiments.09_fine_grained_learning_rate
python -m experiments.10_rmsprop_beta_analysis
python -m experiments.11_rmsprop_learning_rate_analysis
```

Generated plots and CSV files are saved in the `results/` directory.

## Testing

Run the unit tests with:

```bash
python -m pytest -v
```

The tests cover gradient calculations, optimizer updates, numerical stability, and optimization behavior.

## Mathematical Update Rules

**Gradient Descent**

\[
\theta_{t+1}=\theta_t-\eta\nabla f(\theta_t)
\]

**Momentum**

\[
v_t=\beta v_{t-1}+\nabla f(\theta_t)
\]

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

**RMSProp**

\[
s_t=\beta s_{t-1}+(1-\beta)g_t^2
\]

\[
\theta_{t+1}=\theta_t-
\frac{\eta g_t}{\sqrt{s_t}+\epsilon}
\]

**Adam**

Adam combines exponentially weighted first and second gradient moments, with bias correction.

## Research Questions

This project investigates questions such as:

- How does the learning rate affect convergence and numerical stability?
- When does momentum improve optimization compared with vanilla gradient descent?
- How sensitive are adaptive optimizers to their hyperparameters?
- How does feature scaling influence gradient-based optimization?
- How does performance vary across different initial conditions?
- How do batch size and gradient noise affect optimization?

## Experimental Methodology

For meaningful comparisons, experiments use controlled settings wherever appropriate:

- Identical initial points when comparing optimizers.
- Fixed random seeds for reproducibility.
- Multiple initial points to study variation in performance.
- Explicit learning rates, hyperparameters, and iteration budgets.
- Gradient-based stopping criteria.
- Numerical failure tracking and final-loss comparisons.

Results are interpreted within the tested objectives and hyperparameter ranges. No optimizer is assumed to be universally superior.

## Limitations

This is an educational and research-oriented implementation rather than a production optimization library.

The experiments use selected mathematical objectives and synthetic datasets. They do not cover every optimizer variant, learning-rate scheduler, gradient clipping method, sparse optimization technique, or large-scale distributed training setup.

## Future Work

Potential extensions include:

- Convergence-rate analysis.
- Optimization under noisy and stochastic gradients.
- Additional objective functions and test problems.
- More systematic hyperparameter searches.
- Theoretical analysis of convergence and generalization.

## References

References and additional reading will be documented in the research report.
