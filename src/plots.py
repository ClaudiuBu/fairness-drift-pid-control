import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(log_fairness, log_accuracy, log_control, filename_prefix='fig', year_transitions=None):
    """
    Plot experiment results.
    
    Args:
        log_fairness: dict with fairness metrics for each method
        log_accuracy: dict with accuracy metrics for each method
        log_control: list of control signals
        filename_prefix: prefix for saved figure filenames
        year_transitions: list of time steps where years changed (for temporal mode)
    """
    # Create output directory based on filename prefix
    if 'folktables' in filename_prefix:
        output_dir = 'results/folktables/single_run'
    else:
        output_dir = 'results/synthetic/single_run'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- GRAFIC 1: FAIRNESS ---
    plt.figure(figsize=(12, 6))
    
    # Plot all available methods
    colors = {'base': 'grey', 'baseline': 'grey', 'static': 'blue', 
              'sliding': 'orange', 'pid': 'green'}
    styles = {'base': '--', 'baseline': '--', 'static': '-.', 
              'sliding': '-', 'pid': '-'}
    alphas = {'base': 0.5, 'baseline': 0.5, 'static': 0.6, 
              'sliding': 0.8, 'pid': 1.0}
    widths = {'base': 1, 'baseline': 1, 'static': 1, 
              'sliding': 2, 'pid': 3}
    
    first_vals = np.asarray(next(iter(log_fairness.values())))
    series_len = first_vals.shape[1] if first_vals.ndim == 2 else len(first_vals)

    for method, values in log_fairness.items():
        label = method.replace('_', ' ').title()
        values = np.asarray(values)
        if values.ndim == 2:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            plt.plot(mean, label=label, 
                    color=colors.get(method, 'black'),
                    linestyle=styles.get(method, '-'),
                    alpha=alphas.get(method, 0.8),
                    linewidth=widths.get(method, 1))
            plt.fill_between(range(len(mean)), mean - std, mean + std,
                             color=colors.get(method, 'black'), alpha=0.2)
        else:
            plt.plot(values, label=label, 
                    color=colors.get(method, 'black'),
                    linestyle=styles.get(method, '-'),
                    alpha=alphas.get(method, 0.8),
                    linewidth=widths.get(method, 1))
    
    plt.axhline(0, color='black', linestyle=':', label='Target (0.0)', alpha=0.5)
    
    # Add year transitions for temporal mode
    if year_transitions:
        for i, trans in enumerate(year_transitions):
            label_text = 'Year Transition' if i == 0 else None
            plt.axvline(trans, color='red', linestyle='--', alpha=0.6, 
                       linewidth=2, label=label_text)
    # Add drift start line if data looks synthetic (drift at t=15)
    elif series_len >= 20:
        plt.axvline(15, color='red', linestyle=':', label='Drift Start', alpha=0.5)
    
    plt.title("Fairness Drift Mitigation Comparison")
    plt.xlabel("Time Steps")
    plt.ylabel("Demographic Parity Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_fairness.png', dpi=150)
    plt.show()

    # --- GRAFIC 2: ACCURACY ---
    plt.figure(figsize=(12, 5))
    
    first_acc_vals = np.asarray(next(iter(log_accuracy.values())))
    acc_series_len = first_acc_vals.shape[1] if first_acc_vals.ndim == 2 else len(first_acc_vals)

    for method, values in log_accuracy.items():
        label = method.replace('_', ' ').title()
        values = np.asarray(values)
        if values.ndim == 2:
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            plt.plot(mean, label=label,
                    color=colors.get(method, 'black'),
                    linestyle=styles.get(method, '-'),
                    alpha=alphas.get(method, 0.8),
                    linewidth=widths.get(method, 1))
            plt.fill_between(range(len(mean)), mean - std, mean + std,
                             color=colors.get(method, 'black'), alpha=0.2)
        else:
            plt.plot(values, label=label,
                    color=colors.get(method, 'black'),
                    linestyle=styles.get(method, '-'),
                    alpha=alphas.get(method, 0.8),
                    linewidth=widths.get(method, 1))
    
    # Add year transitions
    if year_transitions:
        for trans in year_transitions:
            plt.axvline(trans, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    elif acc_series_len >= 20:
        plt.axvline(15, color='red', linestyle=':', label='Drift Start', alpha=0.5)
    
    plt.title("Impact on Accuracy")
    plt.xlabel("Time Steps")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_accuracy.png', dpi=150)
    plt.show()

    # --- GRAFIC 3: CONTROL SIGNAL ---
    if not log_control:
        return
    
    plt.figure(figsize=(10, 4))
    plt.plot(log_control, color='purple', linewidth=1.5, label='PID Control Signal')
    
    # Add year transitions
    if year_transitions:
        for trans in year_transitions:
            plt.axvline(trans, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    elif len(log_control) >= 20:
        plt.axvline(15, color='red', linestyle=':', alpha=0.5)
    
    plt.title("PID Control Signal Activity")
    plt.xlabel("Time Steps")
    plt.ylabel("Correction Magnitude (u)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_control.png', dpi=150)
    plt.show()

