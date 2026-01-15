from src.experiment import run_experiment
from src.plots import plot_results
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    print("Running PREQUENTIAL Experiment: Baseline, Static GBR, Sliding GBR, PID...")
    log_fairness, log_accuracy, log_control = run_experiment(T=60)
    plot_results(log_fairness, log_accuracy, log_control)
    print("Done! Check the generated PNG files.")
