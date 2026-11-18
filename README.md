# Gradient Descent and Optimization Algorithms from Scratch

A research-oriented educational project implementing common optimization algorithms using **Python and NumPy**, without machine-learning optimizer libraries.

## Algorithms
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD
- Momentum
- RMSProp
- Adam

## Objectives
- Quadratic function: `f(x) = sum(x_i^2)`
- Two-dimensional bowl: `f(x, y) = x^2 + y^2`
- Rosenbrock function: `f(x, y) = (1-x)^2 + 100(y-x^2)^2`
- Linear-regression mean squared error

## Structure
```text
gradient-descent-from-scratch/
├── src/                 # objectives, gradients, optimizers, data, plots
├── experiments/         # six reproducible experiments
├── tests/               # unit tests
├── results/              # generated figures and CSV files
├── report/               # research report template
├── requirements.txt
└── README.md
```

## Setup
Python 3.9+ recommended.

```bash
python -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, or on Windows with `.venv\\Scripts\\activate`.

```bash
pip install -r requirements.txt
```

## Run experiments
Run commands from the project root:

```bash
python -m experiments.01_quadratic_gradient_descent
python -m experiments.02_compare_optimizers_rosenbrock
python -m experiments.03_linear_regression_batch_vs_sgd
python -m experiments.04_learning_rate_study
python -m experiments.05_feature_scaling
python -m experiments.06_adam_learning_rate_study
```

Plots and CSV files are saved in `results/`.

## Run tests
```bash
pytest -v
```

## Update rules
Gradient Descent: `theta_(t+1) = theta_t - eta * gradient_t`.

Momentum: `v_t = beta*v_(t-1) + gradient_t`; then `theta_(t+1) = theta_t - eta*v_t`.

RMSProp: `s_t = beta*s_(t-1) + (1-beta)*gradient_t^2`; scale the gradient by `sqrt(s_t)+epsilon`.

Adam combines exponentially weighted first and second moments with bias correction.

## Research questions
- How does learning rate affect convergence and divergence?
- When does momentum help over vanilla gradient descent?
- How do adaptive methods behave in a narrow curved valley?
- How does feature scaling change optimization?
- How do batch size and gradient noise affect convergence?

## Experimental practice
Use identical objectives and starting points when comparing algorithms. State initialization, learning rate, iteration budget, stopping criterion, and random seed. Repeat stochastic experiments with multiple seeds before making broad claims. Optimizer performance depends on the problem and hyperparameters; no method is universally best.

## Limitations
This is an educational implementation, not a production optimization library. It omits schedulers, gradient clipping, sparse updates, and distributed training.
