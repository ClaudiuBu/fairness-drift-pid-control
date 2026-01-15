import matplotlib.pyplot as plt

def plot_results(log_fairness, log_accuracy, log_control):
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- GRAFIC 1: FAIRNESS ---
    plt.figure(figsize=(12, 6))
    
    # Plotăm toate cele 4 curbe
    plt.plot(log_fairness["base"], label='Baseline', color='grey', linestyle='--', alpha=0.5)
    plt.plot(log_fairness["static"], label='Static GBR (t=0)', color='blue', linestyle='-.', alpha=0.6)
    plt.plot(log_fairness["sliding"], label='Sliding GBR', color='orange', linestyle='-', linewidth=2, alpha=0.8)
    plt.plot(log_fairness["pid"], label='PID Control', color='green', linewidth=3)
    
    plt.axhline(0, color='black', linestyle=':', label='Target (0.0)')
    plt.axvline(15, color='red', linestyle=':', label='Drift Start')
    
    plt.title("Fairness Drift Mitigation: PID vs Sliding Window vs Static")
    plt.xlabel("Time Steps")
    plt.ylabel("Demographic Parity Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1_fairness_comparison.png')
    plt.show()

    # --- GRAFIC 2: ACCURACY ---
    plt.figure(figsize=(12, 5))
    plt.plot(log_accuracy["base"], label='Baseline', color='grey', linestyle='--', alpha=0.3)
    plt.plot(log_accuracy["static"], label='Static', color='blue', alpha=0.3)
    plt.plot(log_accuracy["sliding"], label='Sliding Window', color='orange', alpha=0.6)
    plt.plot(log_accuracy["pid"], label='PID Control', color='green', linewidth=2)
    
    plt.axvline(15, color='red', linestyle=':', label='Drift Start')
    plt.title("Impact on Accuracy")
    plt.xlabel("Time Steps")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig2_accuracy_comparison.png')
    plt.show()

    # --- GRAFIC 3: CONTROL SIGNAL ---
    plt.figure(figsize=(10, 4))
    plt.plot(log_control, color='purple', linewidth=1.5)
    plt.axvline(15, color='red', linestyle=':')
    plt.title("PID Control Signal Activity")
    plt.ylabel("Correction Magnitude (u)")
    plt.tight_layout()
    plt.savefig('fig3_pid_control.png')
    plt.show()

