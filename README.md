# Discovering Hidden Physics

### Hybrid System Identification for Scientific Machine Learning

Master's thesis project by Marcus Galea Jacobsen

---

## Project Summary

This repository contains the code, experiments, and results for a master's thesis on discovering unknown dynamics from partially known physical systems.

The core idea is to combine:

- known mechanistic equations (white-box physics), and
- learned surrogate terms (gray-box modeling),

to recover hidden reaction terms and improve predictive performance.

The implementation is centered around Julia and the SciML ecosystem, with experiments spanning:

- Lotka-Volterra systems
- Enzyme dynamics and hydrolysis kinetics
- Sparse symbolic recovery (SINDy-style)
- Neural differential equation baselines

# Core package code

- [src/hybrid_model.jl](src/hybrid_model.jl): Hybrid model definition and simulation utilities for combining known ODEs with unknown surrogate dynamics.
    - `HybridModel` is the main struct representing the combined system.
    - `HybridPEProblem` is the problem definition for parameter estimation.
- [src/SINDy_methods.jl](src/SINDy_methods.jl): Sparse and reaction-network-oriented discovery utilities.
- [src/polyopt.jl](src/polyopt.jl): Optimization helpers for polynomial/sparse model fitting.
- [src/plot_functions.jl](src/plot_functions.jl): Plotting utilities used across experiments.
## High level diagram of core package 
![framework](https://github.com/MarcusGalea/discovering_hidden_physics/blob/main/plots/framework.JPG)

# Experiments and scripts

- [scripts/LV](scripts/LV): Lotka-Volterra data generation, hidden-model setup, sparse regression, and UDE experiments.
- [scripts/EnzymeDynamics](scripts/EnzymeDynamics): Enzyme benchmark workflows.
- [scripts/Hydrolysis](scripts/Hydrolysis): Hydrolysis benchmark workflows and hyperparameter studies.
- [scripts/structural_identifiability](scripts/structural_identifiability): Notebooks for identifiability analysis.
- [scripts/testing.jl](scripts/testing.jl): Early integrated testing/prototyping script.
## Some plots showing results from the experiments 
![Loss Trace](https://github.com/MarcusGalea/discovering_hidden_physics/blob/main/models/Hydrolysis/Reg/plots/loss_trace_2025-08-15_12-00-54.png)
![Parameter Trace](https://github.com/MarcusGalea/discovering_hidden_physics/blob/main/models/Hydrolysis/Reg/plots/param_trace_2025-08-15_12-00-54.png)




## Methodology

1. Define a partially known ODE system using ModelingToolkit.
2. Split dynamics into known terms and unknown terms.
3. Represent unknown terms using either:
   - sparse symbolic reaction terms, or
   - neural surrogate models.
4. Fit parameters to observational data with SciML optimization tools.
5. Compare discovered dynamics against synthetic or benchmark ground truth.
6. Save fitted parameters, traces, and plots for analysis.

---

## Reproducibility Notes

- Top-level environment files:
  - [Project.toml](Project.toml)
  - [Manifest.toml](Manifest.toml)
- Some subprojects have independent environments (for example [DataDrivenConstrained](DataDrivenConstrained), [NeuralODEs](NeuralODEs), and [UDEs](UDEs)).
- Saved artifacts and traces are stored under [models](models) and [plots](plots).

---