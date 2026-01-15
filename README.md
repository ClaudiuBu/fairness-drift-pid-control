# Fairness Drift Mitigation via PID Control

This repository contains the reference implementation for the paper:

**Adaptive Fairness Control: Mitigating Algorithmic Drift in Non-Stationary Environments using PID**

## Overview
Fairness in deployed machine learning systems is not static and may degrade over time due to distributional and concept drift.
This work proposes a closed-loop mitigation framework that treats fairness as a dynamically regulated signal.
A Proportional–Integral–Derivative (PID) controller monitors demographic parity violations and adaptively reweights training samples during incremental learning.

## Repository Structure
src/
├── data.py # Synthetic data stream generation with fairness drift
├── model.py #  SGDClassifier model
├── fairness.py # Fairness metrics (Demographic Parity)
├── pid.py # PID controller implementation
├── experiment.py # Experimental logic and baselines
├── plots.py # Visualization utilities
run.py # Main entry point
batch_runner.py # Monte Carlo simulation


## Running the Experiments
To reproduce the experiments reported in the paper, run:
1.python3 run.py
2.python3 batch_runner.py

## Reproducibility
All experiments are deterministic given fixed random seeds.
Hyperparameters and experimental settings are documented in the source code.

## License
This code is released for academic and research purposes.