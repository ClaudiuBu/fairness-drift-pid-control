# PID Control for Fairness Drift (POC)

Reference implementation for **Adaptive Fairness Control: Mitigating Algorithmic Drift in Non-Stationary Environments using PID**.

## Overview
This project studies fairness drift in streaming / incremental learning and mitigates it using a **PID controller** that regulates demographic parity violations by adaptively reweighting incoming samples.

## Repository Structure

```text
.
├── src/
│   ├── data.py               # Data loading & stream preparation
│   ├── model.py              # Incremental classifier utilities
│   ├── fairness.py           # Fairness metrics (Demographic Parity)
│   ├── pid.py                # PID controller
│   ├── experiment.py         # Experiment logic and baselines
│   ├── experiment_runner.py  # Orchestrates runs from config
│   ├── benchmark_runner.py   # Static benchmark on Folktables
│   └── plots.py              # Visualization utilities
├── configs/                  # YAML configs (synthetic, folktables)
├── data/                     # Local datasets (not tracked in git)
├── results/                  # MLflow outputs and artifacts
├── requirements.txt
└── README.md
```

## Setup
1. Create and activate a Python environment.
2. Install dependencies from [requirements.txt](requirements.txt).

## Running Experiments
Run experiments via config files:
- Synthetic stream:
	python src/experiment_runner.py configs/synthetic.yaml
- Folktables stream:
	python src/experiment_runner.py configs/folktables.yaml

The runner writes outputs under results/ and also stores the resolved config snapshot.

## Notes
- Large datasets under data/ are intentionally excluded from git.
- Results are logged under results/ (including MLflow runs).

## Reproducibility
Experiments are deterministic given fixed seeds. All hyperparameters are defined in the config YAMLs.

## License
Released for academic and research purposes.