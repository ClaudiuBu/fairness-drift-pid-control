"""
Modular experiment runner using Strategy pattern.

Benefits:
  - No code duplication in predict-log-fit cycles
  - Easy to add new fairness strategies
  - Clean separation of concerns
  - Testable components
"""

import numpy as np
from typing import Dict, List, Tuple
from src.data import generate_batch
from src.model import OnlineModel
from src.fairness import demographic_parity, precision_score, recall_score, f1_score
from src.pid import PIDController
from src.strategies.base import Strategy, BaselineStrategy
from src.strategies.gbr import StaticGBRStrategy, SlidingGBRStrategy
from src.strategies.pid import PIDControlStrategy
from src.core.experiment_utils import (
    compute_method_metrics, log_metrics_to_mlflow, print_experiment_summary
)


class StreamingExperiment:
    """Modular streaming fairness experiment."""
    
    def __init__(self, strategies: Dict[str, Strategy], 
                 log_results: bool = True):
        """
        Args:
            strategies: Dict[name] -> Strategy instance
            log_results: Whether to log metrics to MLflow
        """
        self.strategies = strategies
        self.log_results = log_results
        self.models = {name: OnlineModel() for name in strategies.keys()}
    
    def _evaluate_and_update(self, strategy_name: str, 
                            X: np.ndarray, y: np.ndarray, A: np.ndarray,
                            logs: Dict) -> Tuple[float, float, np.ndarray]:
        """Unified predict-log-fit cycle for any strategy.
        
        Returns:
            (fairness_metric, accuracy_metric, predictions)
        """
        model = self.models[strategy_name]
        strategy = self.strategies[strategy_name]
        
        # 1. PREDICT
        y_pred = model.predict(X)
        
        # 2. LOG
        fairness = demographic_parity(y_pred, A)
        accuracy = (y_pred == y).mean()
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        logs['fairness'][strategy_name].append(fairness)
        logs['accuracy'][strategy_name].append(accuracy)
        logs['precision'][strategy_name].append(precision)
        logs['recall'][strategy_name].append(recall)
        logs['f1'][strategy_name].append(f1)
        
        # 3. CONTROL (if applicable)
        if isinstance(strategy, PIDControlStrategy):
            u = strategy.compute_control_action(fairness)
            logs['control'].append(u)

        # 4. UPDATE
        strategy.fit(model, X, y, A)
        
        return fairness, accuracy, y_pred
    
    def run_streaming(self, T: int = 60, 
                     drift_start: int = 15, 
                     drift_slope: float = 0.08,
                     batch_size: int = 500,
                     verbose: bool = True) -> Dict:
        """Run prequential streaming experiment.
        
        Args:
            T: Number of time steps
            drift_start: When to start concept drift
            drift_slope: Strength of drift per time step
            batch_size: Samples per batch
            verbose: Print progress
        
        Returns:
            Dict with 'fairness', 'accuracy', 'control' logs
        """
        # Initialize logs
        logs = {
            'fairness': {name: [] for name in self.strategies.keys()},
            'accuracy': {name: [] for name in self.strategies.keys()},
            'precision': {name: [] for name in self.strategies.keys()},
            'recall': {name: [] for name in self.strategies.keys()},
            'f1': {name: [] for name in self.strategies.keys()},
            'control': []
        }
        
        # Warm-up batch
        X0, y0, A0 = generate_batch(n=1000, drift_strength=0.0)
        for strategy_name in self.strategies.keys():
            self.strategies[strategy_name].initialize(y0, A0)
            self.strategies[strategy_name].fit(self.models[strategy_name], X0, y0, A0)
        
        if verbose:
            print(f"✓ Warm-up complete: {len(X0)} samples")
        
        # Main loop
        for t in range(T):
            drift = 0.0
            if t >= drift_start:
                drift = drift_slope * (t - drift_start)
            
            X, y, A = generate_batch(n=batch_size, drift_strength=drift)
            
            # Evaluate each strategy
            current_fairness = {}
            for strategy_name in self.strategies.keys():
                fairness, acc, _ = self._evaluate_and_update(
                    strategy_name, X, y, A, logs
                )
                current_fairness[strategy_name] = fairness
                
                # Update sliding history if needed
                if hasattr(self.strategies[strategy_name], "update_history"):
                    self.strategies[strategy_name].update_history(y, A)
            
            if verbose and (t + 1) % max(1, T // 5) == 0:
                print(f"  [{t+1}/{T}] Base DP={abs(current_fairness['base']):.4f}, "
                      f"PID DP={abs(current_fairness.get('pid', 0.0)):.4f}")
        
        self.logs = logs
        return logs
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Compute aggregated metrics across experiment."""
        return compute_method_metrics(
            self.logs['fairness'],
            self.logs['accuracy']
        )


class RobustnessExperiment:
    """Run experiment across multiple random seeds for robustness analysis."""
    
    def __init__(self, num_seeds: int = 20):
        self.num_seeds = num_seeds
        self.all_results = []
    
    def run(self, strategies: Dict[str, Strategy],
            T: int = 60, verbose: bool = True) -> Dict:
        """Run streaming experiment multiple times with different seeds.
        
        Returns:
            Dict with aggregated results across seeds
        """
        results_fairness = {name: np.zeros((self.num_seeds, T)) for name in strategies.keys()}
        results_accuracy = {name: np.zeros((self.num_seeds, T)) for name in strategies.keys()}
        
        seeds = np.random.randint(0, 10000, size=self.num_seeds)
        
        for seed_idx, seed in enumerate(seeds):
            np.random.seed(seed)
            
            exp = StreamingExperiment(strategies, log_results=False)
            logs = exp.run_streaming(T=T, verbose=False)
            
            for method_name in strategies.keys():
                results_fairness[method_name][seed_idx] = np.abs(logs['fairness'][method_name])
                results_accuracy[method_name][seed_idx] = logs['accuracy'][method_name]
            
            self.all_results.append(logs)
            
            if verbose:
                print(f"  [Run {seed_idx+1}/{self.num_seeds}] Seed {seed} completed")
        
        return {
            'fairness': results_fairness,
            'accuracy': results_accuracy,
            'individual_runs': self.all_results
        }


# ============================================================================
# Example: How to use the modular design
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODULAR EXPERIMENT DESIGN - EXAMPLE")
    print("="*70)
    
    # 1. Define strategies
    pid = PIDController(kp=10.0, ki=1.0, kd=0.5, target=0.0)
    strategies = {
        'base': BaselineStrategy(),
        'pid': PIDControlStrategy(pid),
    }
    
    # 2. Run single streaming experiment
    print("\n[1] Single Streaming Experiment")
    exp = StreamingExperiment(strategies)
    logs = exp.run_streaming(T=30, verbose=True)
    print_experiment_summary(logs['fairness'], logs['accuracy'], 
                            title="Single Run Results")
    
    # 3. Log to MLflow (if available)
    try:
        import mlflow
        mlflow.set_experiment("modular_test")
        with mlflow.start_run():
            metrics = compute_method_metrics(logs['fairness'], logs['accuracy'])
            log_metrics_to_mlflow(metrics)
            print("✓ Metrics logged to MLflow")
    except ImportError:
        print("(MLflow not available)")
    
    # 4. Run robustness analysis
    print("\n[2] Robustness Analysis (5 seeds)")
    robust_exp = RobustnessExperiment(num_seeds=5)
    results = robust_exp.run(strategies, T=30, verbose=True)
    
    # Print robustness summary
    print("\n" + "="*70)
    print("  ROBUSTNESS SUMMARY (5 seeds)")
    print("="*70)
    for method in strategies.keys():
        fairness_values = results['fairness'][method]
        mean_fairness = np.mean(np.abs(fairness_values))
        std_fairness = np.std(np.abs(fairness_values.mean(axis=1)))
        print(f"  {method.upper():10s}: DP = {mean_fairness:.4f} ± {std_fairness:.4f}")
    
    print("\n✓ All experiments completed successfully!")
    print("\nBenefits of modular design:")
    print("  ✓ No code duplication in predict-log-fit cycles")
    print("  ✓ Easy to add new strategies (reweighting, post-processing, etc.)")
    print("  ✓ Easy to test individual components")
    print("  ✓ Easy to extend with new fairness metrics")
    print("  ✓ SOTA methods can be added by creating new Strategy subclass")
