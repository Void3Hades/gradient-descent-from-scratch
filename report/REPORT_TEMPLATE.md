# Gradient Descent and Optimization Algorithms from Scratch

**Author:** Rishabh
**Institution:** Indian Institute of Science Education and Research (IISER) Kolkata
**Project:** Gradient Descent and Optimization Algorithms from Scratch
**Tools:** Python, NumPy, Matplotlib, CSV
**Repository:** https://github.com/Void3Hades/gradient-descent-from-scratch

---

## Abstract

This project implements and experimentally investigates several gradient-based optimization algorithms using Python and NumPy. The algorithms include Batch Gradient Descent, Stochastic Gradient Descent (SGD), Mini-batch SGD, Momentum, RMSProp, and Adam.

Experiments are conducted on quadratic functions, the Rosenbrock function, and linear regression problems. The study examines the effects of learning rates, optimizer hyperparameters, feature scaling, and initial conditions on optimization performance.

Particular attention is given to convergence behavior, numerical stability, final objective values, and consistency across multiple initializations.

**Main findings:** [Summarize three or four important observations from the experiments.]

---

## 1. Introduction

Numerical optimization is an important component of machine learning. Many learning algorithms involve minimizing an objective function by iteratively updating model parameters.

Gradient-based methods use information about the gradient of an objective function to determine the direction of parameter updates.

However, the performance of these methods depends on several factors, including:

- Learning rate
- Choice of optimization algorithm
- Initial parameter values
- Curvature of the objective function
- Feature scaling
- Stochasticity in gradient estimates

The objective of this project is to implement common optimization algorithms from scratch and investigate their behavior through controlled experiments.

### 1.1 Project Objectives

1. Implement commonly used gradient-based optimization algorithms.
2. Understand the mathematical principles behind their update rules.
3. Compare their performance on different objective functions.
4. Investigate learning-rate and hyperparameter sensitivity.
5. Examine convergence behavior across multiple initial conditions.
6. Develop reproducible experimental practices.

### 1.2 Research Questions

- How does the learning rate affect convergence and numerical stability?
- Under what conditions does Momentum improve upon standard Gradient Descent?
- How sensitive are RMSProp and Adam to their hyperparameters?
- How does feature scaling influence optimization?
- Do optimizer comparisons remain consistent across different initial conditions?
- Can a low final objective value occur without satisfying a strict gradient-based convergence criterion?

---

## 2. Mathematical Background

### 2.1 Objective Functions

An optimization problem can be written as:

\[
\min_{\theta\in\mathbb{R}^d} f(\theta)
\]

where:

- \(\theta\) represents the parameters.
- \(f(\theta)\) is the objective function.
- \(d\) is the number of parameters.

### 2.2 Gradients

The gradient is the vector of partial derivatives:

\[
\nabla f(\theta)=
\begin{bmatrix}
\frac{\partial f}{\partial\theta_1}\\
\vdots\\
\frac{\partial f}{\partial\theta_d}
\end{bmatrix}
\]

It provides local information about the direction of steepest increase.

### 2.3 Learning Rate

The learning rate, denoted by \(\eta\), controls the size of parameter updates.

A learning rate that is too large may cause divergence or numerical instability, whereas a very small learning rate may lead to slow progress.

### 2.4 Convergence and Stopping Criteria

In this project, gradient-based convergence is assessed using:

\[
\|\nabla f(\theta)\|_2 < \varepsilon
\]

where \(\varepsilon\) is the specified tolerance.

For experiments using the Rosenbrock function, the tolerance is:

\[
\varepsilon=10^{-8}
\]

A low objective value alone does not necessarily imply that this stopping criterion has been satisfied.

### 2.5 Evaluation Metrics

The following metrics are used:

- **Final loss:** Objective-function value at the final parameters.
- **Gradient norm:** Euclidean norm of the final gradient.
- **Iterations:** Number of optimization updates.
- **Convergence rate:** Fraction of runs satisfying the stopping criterion.
- **Numerical failures:** Runs that encounter numerical or validation errors.
- **Mean and median final loss:** Measures of average and typical performance across runs.

---

## 3. Optimization Algorithms

### 3.1 Batch Gradient Descent

\[
\theta_{t+1}=\theta_t-\eta\nabla f(\theta_t)
\]

All training observations are used to calculate the gradient at each update.

### 3.2 Stochastic Gradient Descent

SGD updates parameters using a randomly selected observation or a small sample.

\[
\theta_{t+1}=\theta_t-\eta g_t
\]

Here, \(g_t\) is a stochastic estimate of the gradient.

### 3.3 Mini-batch Gradient Descent

Mini-batch SGD calculates gradients using a subset of the training dataset at each iteration.

### 3.4 Momentum

\[
v_t=\beta v_{t-1}+g_t
\]

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

Momentum accumulates information from previous gradients to influence the current update.

### 3.5 RMSProp

\[
s_t=\beta s_{t-1}+(1-\beta)g_t^2
\]

\[
\theta_{t+1}=
\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
\]

RMSProp maintains an exponentially weighted average of squared gradients to adapt parameter-wise update sizes.

### 3.6 Adam

Adam combines estimates of the first and second moments of gradients:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

With bias corrections:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

The parameter update is:

\[
\theta_{t+1}=
\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

---

## 4. Experimental Setup

### 4.1 Software and Environment

| Component | Implementation |
|---|---|
| Programming language | Python |
| Numerical computation | NumPy |
| Visualization | Matplotlib |
| Data storage | CSV |
| Testing | pytest |
| Development environment | Python virtual environment |

### 4.2 Objective Functions

The experiments use the following objectives.

**Quadratic function:**

\[
f(x)=\sum_{i=1}^{d}x_i^2
\]

**Two-dimensional quadratic:**

\[
f(x,y)=x^2+y^2
\]

**Rosenbrock function:**

\[
f(x,y)=(1-x)^2+100(y-x^2)^2
\]

Its global minimum is at \((1,1)\), where the objective value is zero.

**Linear regression objective:**

\[
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}
(x_i^T\theta-y_i)^2
\]

[Verify the precise MSE convention used in the implementation before final submission.]

### 4.3 Reproducibility

The multi-seed experiments use:

- Random seed: 42
- Number of initial points: 20
- Initial points sampled uniformly from the specified ranges
- Identical initial points across optimizers within each comparison
- Maximum iteration budget: 20,000
- Gradient tolerance: \(10^{-8}\)

### 4.4 Experimental Fairness

For meaningful comparisons:

- The same initial conditions are used across compared optimizers.
- Each run uses a fresh optimizer instance.
- Hyperparameters are explicitly specified.
- Failed runs are recorded rather than silently discarded.
- Results are interpreted only within the tested settings.

---

## 5. Experimental Results

Include the relevant plots and CSV summaries from the `results/` directory in each subsection.

For every experiment, explain the objective, experimental settings, observations, and limitations.

### 5.1 Experiment 01: Quadratic Gradient Descent

**Objective:** Study basic Gradient Descent on a quadratic function.

Record:

- Initial parameters
- Tested learning rates
- Loss progression
- Final parameter values

**Figure:** `01_quadratic_learning_rates.png`

**Observations:** [Discuss how learning rates affect convergence.]

### 5.2 Experiment 02: Optimizer Comparison on Rosenbrock

**Objective:** Compare Gradient Descent, Momentum, RMSProp, and Adam.

**Figures:**
- `02_rosenbrock_loss.png`
- `02_rosenbrock_paths.png`

**Observations:** [Discuss convergence paths, loss reduction, and optimizer behavior.]

### 5.3 Experiment 03: Batch Gradient Descent vs SGD

**Objective:** Compare batch and stochastic optimization for linear regression.

**Figure:** `03_batch_sgd_comparison.png`

**Metrics:** `03_regression_test_metrics.csv`

**Observations:** [Discuss loss behavior, parameter estimates, and test performance.]

### 5.4 Experiment 04: Learning-Rate Study

**Objective:** Investigate the influence of learning rate on optimization.

**Figure:** `04_learning_rate_study.png`

**Metrics:** `04_learning_rate_metrics.csv`

**Observations:** [Identify stable, slow, or unstable learning-rate settings.]

### 5.5 Experiment 05: Feature Scaling

**Objective:** Examine how feature scaling affects gradient-based optimization.

**Figure:** `05_feature_scaling.png`

**Data:**
- `05_scaled_losses.csv`
- `05_unscaled_losses.csv`

**Observations:** [Compare optimization behavior before and after scaling.]

### 5.6 Experiment 06: Adam Learning-Rate Study

**Objective:** Investigate Adam's behavior under different learning rates.

**Figure:** `06_adam_learning_rates.png`

**Metrics:** `06_adam_metrics.csv`

**Observations:** [Discuss the tested learning rates and final objective values.]

### 5.7 Experiment 07: Multi-Seed Optimizer Comparison

**Objective:** Evaluate optimizer consistency across 20 initial points.

**Settings:**

- Random seed: 42
- 20 initial points
- Maximum steps: 20,000
- Gradient tolerance: \(10^{-8}\)

**Figures:**
- `07_multiseed_final_loss.png`
- `07_multiseed_convergence_rate.png`

**Data:**
- `07_multiseed_individual_results.csv`
- `07_multiseed_summary.csv`

**Observations:**

In the tested configuration, Momentum and Adam converged in all 20 runs. Gradient Descent and RMSProp did not satisfy the specified gradient-norm criterion in any run.

These findings are specific to the tested learning rates, initialization range, objective, and iteration budget.

### 5.8 Experiment 08: Learning-Rate Sensitivity

**Objective:** Investigate how different learning rates influence optimizer performance across multiple initial points.

**Figures:**
- `08_learning_rate_convergence.png`
- `08_learning_rate_final_loss.png`

**Data:**
- `08_learning_rate_individual_results.csv`
- `08_learning_rate_summary.csv`

**Observations:**

Adam achieved convergence in all 20 runs at each of its four tested learning rates. Momentum also achieved full convergence at several tested rates, while some larger settings produced numerical failures.

RMSProp did not satisfy the strict gradient-norm convergence criterion in the tested configurations.

[Discuss the differences between convergence rate and final loss.]

### 5.9 Experiment 09: Fine-Grained Learning-Rate Analysis

**Objective:** Examine narrower learning-rate ranges to investigate optimizer sensitivity.

**Figures:**
- `09_fine_grained_convergence.png`
- `09_fine_grained_final_loss.png`

**Data:**
- `09_fine_grained_individual.csv`
- `09_fine_grained_summary.csv`

**Observations:**

Momentum at a learning rate of 0.0015 converged in all 20 runs, with an average of approximately 2,674 iterations.

Adam at a learning rate of 0.05 also converged in all 20 runs, with an average of approximately 3,035 iterations.

These are successful tested configurations, not universal optimal values.

### 5.10 Experiment 10: RMSProp Beta Analysis

**Objective:** Investigate the effect of RMSProp's decay parameter, beta.

**Settings:**

- Fixed learning rate: 0.0005
- Beta values: 0.5, 0.7, 0.9, 0.99, 0.999
- 20 initial points per beta value

**Figure:** `10_rmsprop_beta_convergence.png`

**Data:**
- `10_rmsprop_beta_individual.csv`
- `10_rmsprop_beta_summary.csv`

**Observations:**

None of the tested beta values achieved convergence under the specified gradient tolerance and iteration budget. The mean final losses were also very similar across these configurations.

This suggests that changing beta alone had limited impact under the chosen learning rate and experimental setup.

### 5.11 Experiment 11: RMSProp Learning-Rate Analysis

**Objective:** Investigate RMSProp learning-rate sensitivity while fixing beta.

**Settings:**

- Fixed beta: 0.9
- Eight learning rates between 0.00001 and 0.001
- 20 initial points per learning rate

**Figure:** `11_rmsprop_lr_convergence.png`

**Data:**
- `11_rmsprop_lr_individual.csv`
- `11_rmsprop_lr_summary.csv`

**Observations:**

Among the tested learning rates, 0.0002 produced the lowest mean final loss, approximately \(9.03\times10^{-6}\).

However, none of the runs satisfied the strict gradient-norm convergence criterion within 20,000 steps.

A low final objective value should therefore not be confused with convergence under the chosen stopping rule.

---

## 6. Discussion

### 6.1 Effect of Learning Rate

[Compare the learning-rate studies across different optimizers.]

### 6.2 Optimizer Stability

[Discuss numerical failures, oscillations, and sensitivity to hyperparameters.]

### 6.3 Effect of Initialization

[Discuss how different starting points influence convergence and final loss.]

### 6.4 Convergence vs Final Objective Value

[Explain why low loss and small gradient norm are different criteria.]

### 6.5 RMSProp Hyperparameters

[Discuss the separate effects of beta and learning rate, noting that the two studies vary one parameter at a time.]

### 6.6 General Observations

[Summarize findings that are supported consistently by the experiments.]

---

## 7. Limitations

The project has several limitations:

1. The experiments use a limited collection of objective functions and synthetic datasets.
2. The hyperparameter ranges are selected manually rather than through exhaustive optimization.
3. The maximum iteration budget may prevent some configurations from reaching the stopping criterion.
4. Results on low-dimensional mathematical objectives may not directly generalize to large machine-learning models.
5. The experiments do not cover every optimizer variant or learning-rate scheduling strategy.
6. Random initialization and a finite number of runs limit the strength of statistical conclusions.

---

## 8. Conclusion

This project implemented and experimentally investigated several gradient-based optimization algorithms.

The experiments demonstrate that optimizer performance depends on the objective function, learning rate, hyperparameters, and initialization.

The results also highlight the importance of evaluating optimization methods using multiple metrics rather than relying only on final objective values.

**Main conclusions:**

1. [Conclusion supported by optimizer comparisons.]
2. [Conclusion supported by learning-rate studies.]
3. [Conclusion supported by multi-seed experiments.]
4. [Conclusion supported by RMSProp hyperparameter analysis.]

---

## 9. Future Work

Possible extensions include:

- Studying convergence rates theoretically.
- Investigating stochastic gradient noise.
- Testing additional objective functions.
- Exploring learning-rate schedules.
- Conducting more systematic hyperparameter searches.
- Comparing optimization behavior on larger machine-learning datasets.
- Investigating the relationship between optimization and generalization.

---

## 10. References

1. Robbins, H. and Monro, S. (1951). A Stochastic Approximation Method. *The Annals of Mathematical Statistics*.
2. Polyak, B. T. (1964). Some Methods of Speeding Up the Convergence of Iteration Methods. *USSR Computational Mathematics and Mathematical Physics*.
3. Tieleman, T. and Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the Gradient by a Running Average of Its Recent Magnitude.
4. Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.
5. [Add any additional sources consulted.]
