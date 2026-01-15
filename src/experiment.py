import numpy as np
from src.data import generate_batch
from src.model import OnlineModel
from src.fairness import demographic_parity
from src.pid import PIDController


def calculate_weights_from_data(y, A):
    """Rate-based group reweighting (GBR).

    Computes weights per sensitive group based on *group-level positive rates*.
    Note: this is NOT the classic Kamiran-Calders 'Reweighing' over (A,y) cells.
    """
    count_A0 = np.sum(A == 0)
    count_A1 = np.sum(A == 1)

    if count_A0 == 0 or count_A1 == 0:
        return {0: 1.0, 1: 1.0}

    # Positive rates per group
    rate0 = y[A == 0].mean() if count_A0 > 0 else 0.0
    rate1 = y[A == 1].mean() if count_A1 > 0 else 0.0

    # If rates are equal (or both 0), do nothing
    if np.isclose(rate0, rate1):
        return {0: 1.0, 1: 1.0}

    # Upweight the group with the lower positive rate (simple heuristic)
    if rate0 < rate1:
        return {0: (rate1 / max(rate0, 1e-6)), 1: 1.0}
    else:
        return {0: 1.0, 1: (rate0 / max(rate1, 1e-6))}


def run_experiment(T=60, drift_start=15, drift_slope=0.08, window_size=1000,
                   pid_kp=10.0, pid_ki=1.0, pid_kd=0.5, pid_u_clip=3.0):
    """Run streaming experiment with prequential evaluation.

    Prequential protocol per time-step:
      (1) predict on current batch, log metrics
      (2) update model using current batch (optionally with weights)
    """
    # Models
    model_base = OnlineModel()
    model_static = OnlineModel()
    model_sliding = OnlineModel()
    model_pid = OnlineModel()

    pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd, target=0.0)

    # Logs
    log_fairness = {"base": [], "static": [], "sliding": [], "pid": []}
    log_accuracy = {"base": [], "static": [], "sliding": [], "pid": []}
    log_control = []

    # --- Warm-up / initialization batch (t=0) ---
    X0, y0, A0 = generate_batch(n=1000, drift_strength=0.0)

    # Initialize all models (no leakage concern at t=0 warm-up)
    model_base.fit(X0, y0)
    model_pid.fit(X0, y0)

    static_weights_map = calculate_weights_from_data(y0, A0)
    w0_static = np.array([static_weights_map[a] for a in A0], dtype=float)
    model_static.fit(X0, y0, sample_weight=w0_static)

    # Sliding starts identical to static; history contains only past data
    model_sliding.fit(X0, y0, sample_weight=w0_static)
    history_y = y0.copy()
    history_A = A0.copy()

    print(f"Static GBR weights at t=0: {static_weights_map}")

    # --- Stream loop ---
    for t in range(T):
        drift = 0.0
        if t >= drift_start:
            drift = drift_slope * (t - drift_start)

        X, y, A = generate_batch(n=500, drift_strength=drift)

        # -------------------------------
        # 1) BASELINE (prequential)
        # -------------------------------
        y_pred = model_base.predict(X)
        log_fairness["base"].append(demographic_parity(y_pred, A))
        log_accuracy["base"].append((y_pred == y).mean())
        model_base.fit(X, y)

        # -------------------------------
        # 2) STATIC GBR (prequential)
        # -------------------------------
        y_pred = model_static.predict(X)
        log_fairness["static"].append(demographic_parity(y_pred, A))
        log_accuracy["static"].append((y_pred == y).mean())

        w_static = np.array([static_weights_map[a] for a in A], dtype=float)
        model_static.fit(X, y, sample_weight=w_static)

        # -------------------------------
        # 3) SLIDING GBR (prequential)
        #   weights computed from *past* window only
        # -------------------------------
        y_pred = model_sliding.predict(X)
        log_fairness["sliding"].append(demographic_parity(y_pred, A))
        log_accuracy["sliding"].append((y_pred == y).mean())

        # [FIX APPLIED HERE]
        # 1. Calculate weights from HISTORY (past data only)
        sliding_weights_map = calculate_weights_from_data(history_y, history_A)
        w_sliding = np.array([sliding_weights_map[a] for a in A], dtype=float)
        
        # 2. Update model using these causal weights
        model_sliding.fit(X, y, sample_weight=w_sliding)

        # 3. Update history AFTER the model update (current batch becomes past)
        history_y = np.concatenate([history_y, y])
        history_A = np.concatenate([history_A, A])
        if len(history_y) > window_size:
            history_y = history_y[-window_size:]
            history_A = history_A[-window_size:]

        # -------------------------------
        # 4) PID Control (prequential)
        # -------------------------------
        y_pred = model_pid.predict(X)
        f_current = demographic_parity(y_pred, A)
        log_fairness["pid"].append(f_current)
        log_accuracy["pid"].append((y_pred == y).mean())

        # Control action computed on current observed fairness
        u = pid.step(f_current)
        u = float(np.clip(u, -pid_u_clip, pid_u_clip))
        log_control.append(u)

        # Stable, symmetric weight mapping
        scale = float(np.exp(-u))  # stable mapping; negative u => upweight A=0
        w_pid = np.ones_like(y, dtype=float)
        w_pid[A == 0] *= scale
        w_pid[A == 1] *= (1.0 / scale)

        model_pid.fit(X, y, sample_weight=w_pid)

    return log_fairness, log_accuracy, log_control


if __name__ == "__main__":
    np.random.seed(42)
    print("Running prequential experiment: Baseline, Static GBR, Sliding GBR, PID...")
    log_fairness, log_accuracy, log_control = run_experiment(T=60)
    from src.plots import plot_results
    plot_results(log_fairness, log_accuracy, log_control)
    print("Done! Check the generated PNG files.")