"""
Run experiment with real-world Folktables data.

This script demonstrates how to use the FolktablesDataStream
for fairness drift experiments with real census data.
"""

import numpy as np
from src.data import FolktablesDataStream, generate_folktables_batch
from src.model import OnlineModel
from src.fairness import demographic_parity
from src.pid import PIDController
from src.plots import plot_results


def run_folktables_experiment(task='income', states=['CA'], years=[2018],
                               mode='temporal', T=60, pid_kp=10.0, pid_ki=1.0, pid_kd=0.5):
    """
    Run streaming experiment with Folktables data.
    
    Args:
        task: 'income' or 'employment'
        states: list of state codes
        years: list of years for temporal drift
        mode: 'temporal', 'static', or 'geographic'
        T: number of time steps
        pid_kp, pid_ki, pid_kd: PID parameters
    """
    print(f"Initializing Folktables stream: task={task}, states={states}, years={years}, mode={mode}")
    
    # Initialize data stream
    stream = FolktablesDataStream(
        task=task,
        states=states,
        years=years,
        sensitive_attribute='SEX',  # Male vs Female
        batch_size=500,
        mode=mode
    )
    
    # Initialize models
    model_base = OnlineModel()
    model_pid = OnlineModel()
    pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd, target=0.0)
    
    # Logging
    log_fairness = {'baseline': [], 'pid': []}
    log_accuracy = {'baseline': [], 'pid': []}
    log_control = []
    year_transitions = []  # Track when years change (for plotting)
    
    print(f"Running {T} time steps...")
    
    # Track current year for temporal mode
    if mode == 'temporal' and hasattr(stream, 'current_year_idx'):
        current_year = stream.year_datasets[stream.current_year_idx]['year']
        print(f"Starting with year: {current_year}\n")
    
    for t in range(T):
        # Track year changes in temporal mode
        if mode == 'temporal' and hasattr(stream, 'current_year_idx'):
            new_year = stream.year_datasets[stream.current_year_idx]['year']
            if t > 0 and new_year != current_year:
                year_transitions.append(t)
                print(f"\n{'='*60}")
                print(f"⚠️  YEAR TRANSITION at step {t}: {current_year} → {new_year}")
                print(f"{'='*60}\n")
                current_year = new_year
        
        # Get batch from stream
        X, y, A = stream.get_batch()
        
        # === BASELINE MODEL ===
        if t > 0:
            y_pred_base = model_base.predict(X)
            dp_base = demographic_parity(y_pred_base, A)
            acc_base = np.mean(y_pred_base == y)
        else:
            dp_base, acc_base = 0.0, 0.5
        
        log_fairness['baseline'].append(dp_base)
        log_accuracy['baseline'].append(acc_base)
        
        # Update baseline (no weights)
        model_base.fit(X, y)
        
        # === PID MODEL ===
        if t > 0:
            y_pred_pid = model_pid.predict(X)
            dp_pid = demographic_parity(y_pred_pid, A)
            acc_pid = np.mean(y_pred_pid == y)
        else:
            dp_pid, acc_pid = 0.0, 0.5
        
        log_fairness['pid'].append(dp_pid)
        log_accuracy['pid'].append(acc_pid)
        
        # PID control
        u = pid.step(dp_pid)
        u = float(np.clip(u, -3.0, 3.0))  # Clip control signal
        log_control.append(u)
        
        # Compute weights (stable exponential mapping, same as experiment.py)
        scale = float(np.exp(-u))  # negative u => upweight A=0
        weights = np.ones(len(y), dtype=float)
        weights[A == 0] *= scale
        weights[A == 1] *= (1.0 / scale)
        
        # Update PID model with weights
        model_pid.fit(X, y, sample_weight=weights)
        
        if (t + 1) % 10 == 0:
            print(f"Step {t+1}/{T} | Baseline DP: {dp_base:.3f}, Acc: {acc_base:.3f} | "
                  f"PID DP: {dp_pid:.3f}, Acc: {acc_pid:.3f}, u: {u:.3f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Show year transitions info
    if mode == 'temporal' and year_transitions:
        print(f"Year transitions occurred at steps: {year_transitions}")
    
    plot_results(log_fairness, log_accuracy, log_control, 
                 filename_prefix=f'folktables_{task}_{mode}',
                 year_transitions=year_transitions if mode == 'temporal' else None)
    
    print(f"Done! Results saved with prefix 'folktables_{task}_{mode}'")
    
    return log_fairness, log_accuracy, log_control


if __name__ == "__main__":
    np.random.seed(42)
    
    # Example 1: Static mode (single year, no real drift)
    # print("\n" + "="*60)
    # print("EXPERIMENT 1: Static Mode (Single Year 2018)")
    # print("="*60)
    # run_folktables_experiment(
    #     task='income', 
    #     states=['CA'], 
    #     years=[2018],
    #     mode='static',
    #     T=60
    # )
    
    # Example 2: Temporal drift (multiple years) - RECOMMENDED!
    print("\n" + "="*60)
    print("EXPERIMENT 2: Temporal Drift (2014-2018)")
    print("="*60)
    print("This simulates REAL temporal drift from census data!")
    # Uncomment to run:
    run_folktables_experiment(
        task='income',
        states=['CA'],
        years=[2014, 2015, 2016, 2017, 2018],
        mode='temporal',
        T=100,
        pid_kp=5.0,  # More conservative for real drift
        pid_ki=0.5,
        pid_kd=0.2
    )
    
    # Example 3: Geographic drift (multiple states)
    # Uncomment to run:
    # print("\n" + "="*60)
    # print("EXPERIMENT 3: Geographic Drift (CA → NY → TX)")
    # print("="*60)
    # run_folktables_experiment(
    #     task='income',
    #     states=['CA', 'NY', 'TX'],
    #     years=[2018],
    #     mode='geographic',
    #     T=90
    # )
