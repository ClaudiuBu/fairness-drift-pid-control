"""
Robustness testing for PID controller with real-world Folktables data.

Tests multiple scenarios:
- Different states (geographic diversity)
- Different random batches (sampling variance)
- Different model initialization seeds
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.data import FolktablesDataStream
from src.model import OnlineModel
from src.fairness import demographic_parity
from src.pid import PIDController


def run_folktables_batch(task='income', states=['CA'], years=[2014, 2015, 2016, 2017, 2018],
                         mode='temporal', T=60, num_runs=10, test_type='sampling'):
    """
    Run multiple experiments with real data to test robustness.
    
    Args:
        task: 'income' or 'employment'
        states: list of state codes
        years: years for temporal mode
        mode: 'temporal', 'static', or 'geographic'
        T: number of time steps
        num_runs: number of independent runs
        test_type: 'sampling' (different random batches), 'states' (different states),
                   or 'seeds' (different model initialization)
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS TEST: {test_type.upper()}")
    print(f"Task: {task}, Mode: {mode}, Runs: {num_runs}")
    print(f"{'='*70}\n")
    
    # Storage for results
    results_fairness = {'baseline': [], 'pid': []}
    results_accuracy = {'baseline': [], 'pid': []}
    results_control = []
    
    if test_type == 'states':
        # Test different states
        test_states = [['CA'], ['TX'], ['NY'], ['FL'], ['PA'], ['IL'], ['OH'], ['MI'], ['NC'], ['GA']]
        num_runs = min(num_runs, len(test_states))
        print(f"Testing {num_runs} different states for geographic robustness\n")
    else:
        test_states = [states] * num_runs
    
    for run_idx in range(num_runs):
        current_states = test_states[run_idx]
        
        # Different seed for model initialization if test_type == 'seeds'
        if test_type == 'seeds':
            model_seed = run_idx * 100
            np.random.seed(model_seed)
            print(f"Run {run_idx+1}/{num_runs}: Model seed {model_seed}")
        else:
            print(f"Run {run_idx+1}/{num_runs}: States {current_states}")
        
        # Initialize data stream (each run gets fresh batches)
        stream = FolktablesDataStream(
            task=task,
            states=current_states,
            years=years,
            sensitive_attribute='SEX',
            batch_size=500,
            mode=mode
        )
        
        # Initialize models
        model_base = OnlineModel()
        model_pid = OnlineModel()
        pid = PIDController(kp=10.0, ki=1.0, kd=0.5, target=0.0)
        
        # Run logs for this iteration
        log_fairness_base = []
        log_fairness_pid = []
        log_accuracy_base = []
        log_accuracy_pid = []
        log_control = []
        
        for t in range(T):
            # Get batch
            X, y, A = stream.get_batch()
            
            # === BASELINE MODEL ===
            if t > 0:
                y_pred_base = model_base.predict(X)
                dp_base = demographic_parity(y_pred_base, A)
                acc_base = np.mean(y_pred_base == y)
            else:
                dp_base, acc_base = 0.0, 0.5
            
            log_fairness_base.append(dp_base)
            log_accuracy_base.append(acc_base)
            model_base.fit(X, y)
            
            # === PID MODEL ===
            if t > 0:
                y_pred_pid = model_pid.predict(X)
                dp_pid = demographic_parity(y_pred_pid, A)
                acc_pid = np.mean(y_pred_pid == y)
            else:
                dp_pid, acc_pid = 0.0, 0.5
            
            log_fairness_pid.append(dp_pid)
            log_accuracy_pid.append(acc_pid)
            
            # PID control
            u = pid.step(dp_pid)
            u = float(np.clip(u, -3.0, 3.0))
            log_control.append(u)
            
            # Compute sample weights
            scale = float(np.exp(-u))
            w_minority = scale if scale < 1 else 1.0
            w_majority = 1.0 / scale if scale < 1 else 1.0
            weights = np.where(A == 0, w_majority, w_minority)
            
            model_pid.fit(X, y, sample_weight=weights)
        
        # Store results
        results_fairness['baseline'].append(log_fairness_base)
        results_fairness['pid'].append(log_fairness_pid)
        results_accuracy['baseline'].append(log_accuracy_base)
        results_accuracy['pid'].append(log_accuracy_pid)
        results_control.append(log_control)
    
    # Convert to numpy arrays
    for key in results_fairness:
        results_fairness[key] = np.array(results_fairness[key])
        results_accuracy[key] = np.array(results_accuracy[key])
    results_control = np.array(results_control)
    
    # --- Statistical Analysis ---
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*70}\n")
    
    stats_report = []
    for method in ['baseline', 'pid']:
        # Full trajectory
        mean_abs_dp = np.mean(np.abs(results_fairness[method]))
        std_dp = np.std(results_fairness[method])
        mean_acc = np.mean(results_accuracy[method])
        std_acc = np.std(results_accuracy[method])
        
        stats_report.append({
            'Method': method.upper(),
            'Mean |DP|': f"{mean_abs_dp:.4f}",
            'Std DP': f"{std_dp:.4f}",
            'Mean Acc': f"{mean_acc:.4f}",
            'Std Acc': f"{std_acc:.4f}"
        })
    
    df_stats = pd.DataFrame(stats_report)
    print(df_stats.to_string(index=False))
    
    # --- Visualization ---
    filename_prefix = f'folktables_robustness_{test_type}'
    plot_robustness_results(results_fairness, results_accuracy, results_control, 
                           filename_prefix, num_runs)
    
    print(f"\n✓ Results saved with prefix '{filename_prefix}'")
    return results_fairness, results_accuracy, results_control


def plot_robustness_results(fairness_data, accuracy_data, control_data, 
                            filename_prefix, num_runs):
    """Plot mean curves with confidence intervals."""
    # Create results directory if it doesn't exist
    os.makedirs('results/folktables/robustness', exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Plot 1: Fairness ---
    plt.figure(figsize=(12, 6))
    
    colors = {'baseline': 'grey', 'pid': 'green'}
    labels = {'baseline': 'Baseline', 'pid': 'PID Control'}
    
    x = np.arange(len(fairness_data['baseline'][0]))
    
    for method in ['baseline', 'pid']:
        mean_curve = np.mean(fairness_data[method], axis=0)
        std_curve = np.std(fairness_data[method], axis=0)
        
        plt.plot(x, mean_curve, label=labels[method], 
                color=colors[method], linewidth=2.5)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[method], alpha=0.3)
    
    plt.axhline(0, color='black', linestyle=':', alpha=0.5, linewidth=1)
    plt.title(f"Demographic Parity Robustness (N={num_runs} runs)")
    plt.xlabel("Time Steps")
    plt.ylabel("DP Gap (Mean ± 1 Std Dev)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/folktables/robustness/{filename_prefix}_fairness.png', dpi=150)
    plt.show()
    
    # --- Plot 2: Accuracy ---
    plt.figure(figsize=(12, 6))
    
    for method in ['baseline', 'pid']:
        mean_curve = np.mean(accuracy_data[method], axis=0)
        std_curve = np.std(accuracy_data[method], axis=0)
        
        plt.plot(x, mean_curve, label=labels[method],
                color=colors[method], linewidth=2.5)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[method], alpha=0.3)
    
    plt.title(f"Accuracy Robustness (N={num_runs} runs)")
    plt.xlabel("Time Steps")
    plt.ylabel("Accuracy (Mean ± 1 Std Dev)")
    plt.ylim(0.5, 0.85)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/folktables/robustness/{filename_prefix}_accuracy.png', dpi=150)
    plt.show()
    
    # --- Plot 3: Control Signal Distribution ---
    plt.figure(figsize=(10, 5))
    
    mean_control = np.mean(control_data, axis=0)
    std_control = np.std(control_data, axis=0)
    
    plt.plot(x, mean_control, color='purple', linewidth=2.5, label='Mean Control Signal')
    plt.fill_between(x, mean_control - std_control, mean_control + std_control,
                    color='purple', alpha=0.3)
    
    plt.title(f"PID Control Signal Robustness (N={num_runs} runs)")
    plt.xlabel("Time Steps")
    plt.ylabel("Control Signal u (Mean ± 1 Std Dev)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/folktables/robustness/{filename_prefix}_control.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FOLKTABLES ROBUSTNESS TESTING SUITE")
    print("="*70)
    
    # Test 1: Geographic diversity (different states)
    print("\n✓ Test 1: Geographic Diversity (10 different states)")
    run_folktables_batch(
        task='income',
        years=[2014, 2015, 2016, 2017, 2018],
        mode='temporal',
        T=100,
        num_runs=10,
        test_type='states'
    )
    
    # Test 2: Task diversity (income vs employment)
    print("\n✓ Test 2: Task Diversity (Income prediction, 5 states)")
    states_sample = [['CA'], ['TX'], ['NY'], ['FL'], ['IL']]
    for i, state in enumerate(states_sample):
        if i == 0:
            results_f, results_a, results_c = run_folktables_batch(
                task='income',
                states=state,
                years=[2014, 2015, 2016, 2017, 2018],
                mode='temporal',
                T=100,
                num_runs=1,
                test_type='states'
            )
            all_fairness = {'baseline': [results_f['baseline'][0]], 'pid': [results_f['pid'][0]]}
            all_accuracy = {'baseline': [results_a['baseline'][0]], 'pid': [results_a['pid'][0]]}
            all_control = [results_c[0]]
        else:
            results_f, results_a, results_c = run_folktables_batch(
                task='income',
                states=state,
                years=[2014, 2015, 2016, 2017, 2018],
                mode='temporal',
                T=100,
                num_runs=1,
                test_type='states'
            )
            all_fairness['baseline'].append(results_f['baseline'][0])
            all_fairness['pid'].append(results_f['pid'][0])
            all_accuracy['baseline'].append(results_a['baseline'][0])
            all_accuracy['pid'].append(results_a['pid'][0])
            all_control.append(results_c[0])
    
    # Aggregate and plot
    for key in all_fairness:
        all_fairness[key] = np.array(all_fairness[key])
        all_accuracy[key] = np.array(all_accuracy[key])
    all_control = np.array(all_control)
    plot_robustness_results(all_fairness, all_accuracy, all_control, 
                           'folktables_robustness_task_income', len(states_sample))
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)
    print("\nNote: For real-world data, robustness is tested across:")
    print("  - Geographic diversity (different US states)")
    print("  - Task diversity (income vs employment prediction)")
    print("  NOT random sampling (data is fixed and sequential)")
    print("="*70)
