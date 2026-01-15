import numpy as np
import matplotlib.pyplot as plt
from src.experiment import run_experiment
import pandas as pd

# --- Configuration ---
NUM_SEEDS = 20
T = 60
DRIFT_START = 15

def run_batch_analysis():
    print(f"Running robustness check over {NUM_SEEDS} random seeds...")
    
    # Storage for results: [method][seed, time_step]
    methods = ["base", "static", "sliding", "pid"]
    results_fairness = {m: np.zeros((NUM_SEEDS, T)) for m in methods}
    results_accuracy = {m: np.zeros((NUM_SEEDS, T)) for m in methods}
    
    # --- Monte Carlo Loop ---
    seeds = np.random.randint(0, 10000, size=NUM_SEEDS)
    
    for i, seed in enumerate(seeds):
        # Reset seed for this run
        np.random.seed(seed)
        
        # Run standard experiment
        # Note: Suppress print outputs from experiment if possible, or ignore them
        log_fairness, log_accuracy, _ = run_experiment(T=T, drift_start=DRIFT_START)
        
        # Store traces
        for m in methods:
            results_fairness[m][i, :] = log_fairness[m]
            results_accuracy[m][i, :] = log_accuracy[m]
            
        print(f"  [Run {i+1}/{NUM_SEEDS}] Seed {seed} completed.")

    # --- Aggregation & Stats ---
    print("\n--- Statistical Report (Post-Drift t > 15) ---")
    stats_report = []
    
    for m in methods:
        # Slice data after drift onset
        post_drift_data = results_fairness[m][:, DRIFT_START:]
        
        # Calculate metrics
        avg_fairness_gap = np.mean(np.abs(post_drift_data)) # Mean Absolute Gap
        std_fairness = np.std(post_drift_data)              # Stability measure
        
        stats_report.append({
            "Method": m,
            "Mean |DP|": avg_fairness_gap,
            "Std Dev (Stability)": std_fairness
        })
        
    df_stats = pd.DataFrame(stats_report)
    print(df_stats.to_string(index=False))
    
    # --- Visualization ---
    plot_confidence_intervals(results_fairness, "Demographic Parity Gap (Mean ± 1 Std Dev)", "fig_robustness_fairness.png")
    plot_confidence_intervals(results_accuracy, "Accuracy (Mean ± 1 Std Dev)", "fig_robustness_accuracy.png")
    print("\nDone. Check 'fig_robustness_fairness.png' and 'fig_robustness_accuracy.png'.")

def plot_confidence_intervals(data_dict, title, filename):
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {'base': 'grey', 'static': 'blue', 'sliding': 'orange', 'pid': 'green'}
    styles = {'base': '--', 'static': '-.', 'sliding': '-', 'pid': '-'}
    labels = {'base': 'Baseline', 'static': 'Static GBR', 'sliding': 'Sliding Window', 'pid': 'PID Control'}
    
    x = np.arange(T)
    
    for m in data_dict:
        # Compute stats along the seed axis (axis 0)
        mean_curve = np.mean(data_dict[m], axis=0)
        std_curve = np.std(data_dict[m], axis=0)
        
        # Plot Mean
        plt.plot(x, mean_curve, label=labels[m], color=colors[m], linestyle=styles[m], linewidth=2)
        
        # Plot Confidence Interval (Shaded)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=colors[m], alpha=0.15)
        
    plt.axvline(DRIFT_START, color='red', linestyle=':', label='Drift Start')
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    
    plt.title(f"Robustness Analysis (N={NUM_SEEDS}): {title}")
    plt.xlabel("Time Steps")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show() # Uncomment if running in a notebook

if __name__ == "__main__":
    # Ensure this is independent of the seed in run.py
    run_batch_analysis()