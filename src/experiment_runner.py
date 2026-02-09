"""
Unified experiment runner with config support.
Consolidates logic from run.py, batch_runner.py, etc.
"""

import yaml
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
import mlflow
from scipy import stats

from src.experiment import run_experiment
from src.data import FolktablesDataStream
from src.model import OnlineModel
from src.fairness import demographic_parity
from src.pid import PIDController
from src.plots import plot_results


class ExperimentRunner:
    """Unified runner for all experiment types."""
    
    def __init__(self, config_path):
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exp_config = self.config['experiment']
        self.data_config = self.config['data']
        self.output_dir = self.config['output']['dir']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log file
        self.log_file = os.path.join(
            self.output_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    def log(self, message):
        """Log to both console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def save_config(self):
        """Save config used for this experiment."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = os.path.join(self.output_dir, f"config_{timestamp}.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
        self.log(f"✓ Config saved: {config_file}")
        

    
    def run_synthetic_single(self):
        """Run single experiment on synthetic data."""
        self.log("\n" + "="*70)
        self.log("SYNTHETIC DATA - SINGLE RUN")
        self.log("="*70)
        
        with mlflow.start_run(run_name=f"synthetic_single_{datetime.now().strftime('%H%M%S')}"):
            np.random.seed(42)
            log_fairness, log_accuracy, log_control = run_experiment(
                T=self.data_config['num_batches']
            )
            
            # Log metrics to MLflow
            for method in log_fairness.keys():
                mean_dp = np.mean(np.abs(log_fairness[method]))
                std_dp = np.std(log_fairness[method])
                mean_acc = np.mean(log_accuracy[method])
                std_acc = np.std(log_accuracy[method])
                
                mlflow.log_metrics({
                    f'{method}_mean_dp': mean_dp,
                    f'{method}_std_dp': std_dp,
                    f'{method}_mean_acc': mean_acc,
                    f'{method}_std_acc': std_acc
                })
        
        output_prefix = os.path.join(self.output_dir, "single_run/fig")
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        plot_results(log_fairness, log_accuracy, log_control, 
                    filename_prefix=output_prefix)
        self.log("✓ Single run complete (metrics logged to MLflow)")
        return log_fairness, log_accuracy, log_control
    
    def run_synthetic_robustness(self):
        """Run robustness analysis on synthetic data."""
        self.log("\n" + "="*70)
        with mlflow.start_run(run_name=f"synthetic_robustness_{datetime.now().strftime('%H%M%S')}"):
            num_seeds = self.config['robustness']['num_seeds']
            T = self.data_config['num_batches']
            methods = self.exp_config.get('methods', ['base', 'static', 'sliding', 'pid'])
            
            results_fairness = {m: np.zeros((num_seeds, T)) for m in methods}
            results_accuracy = {m: np.zeros((num_seeds, T)) for m in methods}
            
            seeds = np.random.randint(0, 10000, size=num_seeds)
            
            for i, seed in enumerate(seeds):
                np.random.seed(seed)
                log_fairness, log_accuracy, _ = run_experiment(T=T)
                
                for m in methods:
                    results_fairness[m][i, :] = log_fairness[m]
                    results_accuracy[m][i, :] = log_accuracy[m]
                
                self.log(f"  [Run {i+1}/{num_seeds}] Seed {seed} completed")
            
            # Calculate statistics
            stats_report = []
            for m in methods:
                mean_dp = np.mean(np.abs(results_fairness[m]))
                std_dp = np.std(results_fairness[m])
                mean_acc = np.mean(results_accuracy[m])
                std_acc = np.std(results_accuracy[m])
                
                stats_report.append({
                    'method': m,
                    'mean_dp': mean_dp,
                    'std_dp': std_dp,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc
                })
                
                # Log to MLflow
                mlflow.log_metrics({
                    f'{m}_mean_dp': mean_dp,
                    f'{m}_std_dp': std_dp,
                    f'{m}_mean_acc': mean_acc,
                    f'{m}_std_acc': std_acc
                })
            
            # Statistical tests: compare PID vs Baseline
            if 'pid' in methods and 'base' in methods:
                pid_dp = np.abs(results_fairness['pid']).flatten()
                base_dp = np.abs(results_fairness['base']).flatten()
                t_stat, p_value = stats.ttest_ind(pid_dp, base_dp)
                mlflow.log_metrics({
                    'pid_vs_base_ttest_pvalue': p_value,
                    'pid_vs_base_ttest_statistic': t_stat
                })
                self.log(f"\n✓ Statistical Test (PID vs Baseline):")
                self.log(f"  t-statistic: {t_stat:.4f}")
                self.log(f"  p-value: {p_value:.6f}")
                self.log(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
            
            # Generate plots
            output_prefix = os.path.join(self.output_dir, "robustness/fig")
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
            plot_results(results_fairness, results_accuracy, None,
                        filename_prefix=output_prefix)
            self.log("✓ Robustness analysis complete (plots saved to " + output_prefix + ")")
        
        return results_fairness, results_accuracy
    
    def run_folktables_single(self):
        """Run single experiment on Folktables data."""
        self.log("\n" + "="*70)
        self.log("FOLKTABLES - SINGLE RUN")
        self.log("="*70)
        
        with mlflow.start_run(run_name=f"folktables_single_{datetime.now().strftime('%H%M%S')}"):
            stream = FolktablesDataStream(
                task=self.data_config['task'],
                states=self.data_config['states'][:1],  # First state only
                years=self.data_config['years'],
                sensitive_attribute=self.data_config['sensitive_attribute'],
                batch_size=self.data_config['batch_size'],
                mode=self.data_config['mode']
            )
            
            model_base = OnlineModel()
            model_pid = OnlineModel()
            pid = PIDController(
                kp=self.config['pid_controller']['kp'],
                ki=self.config['pid_controller']['ki'],
                kd=self.config['pid_controller']['kd'],
                target=self.config['pid_controller']['target']
            )
            
            log_fairness = {'baseline': [], 'pid': []}
            log_accuracy = {'baseline': [], 'pid': []}
            log_control = []
            year_transitions = []
            
            T = self.data_config['num_time_steps']
            current_year = None
            
            for t in range(T):
                # Track year changes
                if hasattr(stream, 'current_year_idx'):
                    new_year = stream.year_datasets[stream.current_year_idx]['year']
                    if t > 0 and new_year != current_year:
                        year_transitions.append(t)
                        self.log(f"  ⚠️  YEAR TRANSITION at step {t}: {current_year} → {new_year}")
                    current_year = new_year
                
                X, y, A = stream.get_batch()
                
                # Baseline
                if t > 0:
                    y_pred_base = model_base.predict(X)
                    dp_base = demographic_parity(y_pred_base, A)
                    acc_base = np.mean(y_pred_base == y)
                else:
                    dp_base, acc_base = 0.0, 0.5
                
                log_fairness['baseline'].append(dp_base)
                log_accuracy['baseline'].append(acc_base)
                model_base.fit(X, y)
                
                # PID
                if t > 0:
                    y_pred_pid = model_pid.predict(X)
                    dp_pid = demographic_parity(y_pred_pid, A)
                    acc_pid = np.mean(y_pred_pid == y)
                else:
                    dp_pid, acc_pid = 0.0, 0.5
                
                log_fairness['pid'].append(dp_pid)
                log_accuracy['pid'].append(acc_pid)
                
                u = pid.step(dp_pid)
                u = float(np.clip(u, -3.0, 3.0))
                log_control.append(u)
                
                scale = float(np.exp(-u))
                w_minority = scale if scale < 1 else 1.0
                w_majority = 1.0 / scale if scale < 1 else 1.0
                weights = np.where(A == 0, w_majority, w_minority)
                
                model_pid.fit(X, y, sample_weight=weights)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'baseline_mean_dp': np.mean(np.abs(log_fairness['baseline'])),
                'baseline_std_dp': np.std(log_fairness['baseline']),
                'baseline_mean_acc': np.mean(log_accuracy['baseline']),
                'baseline_std_acc': np.std(log_accuracy['baseline']),
                'pid_mean_dp': np.mean(np.abs(log_fairness['pid'])),
                'pid_std_dp': np.std(log_fairness['pid']),
                'pid_mean_acc': np.mean(log_accuracy['pid']),
                'pid_std_acc': np.std(log_accuracy['pid']),
            })
            
            output_prefix = os.path.join(self.output_dir, "single_run/folktables_income_temporal")
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
            
            plot_results(log_fairness, log_accuracy, log_control,
                        filename_prefix=output_prefix,
                        year_transitions=year_transitions)
            
            self.log("✓ Single run complete (metrics logged to MLflow)")
        
        return log_fairness, log_accuracy, log_control
    
    def run(self):
        """Main entry point - route to appropriate experiment."""
        # Setup MLflow
        mlflow.set_tracking_uri(os.path.join(self.output_dir, 'mlruns'))
        mlflow.set_experiment(self.exp_config['name'])
        mlflow.end_run()  # End any previous run
        
        self.save_config()
        self.log(f"\n{'='*70}")
        self.log(f"Experiment: {self.exp_config['name']}")
        self.log(f"Type: {self.exp_config['type']}")
        self.log(f"Data: {self.exp_config['data_type']}")
        self.log(f"{'='*70}\n")
        
        exp_type = self.exp_config['type']
        data_type = self.exp_config['data_type']
        
        if data_type == 'synthetic':
            if exp_type == 'single_run':
                return self.run_synthetic_single()
            elif exp_type == 'robustness':
                return self.run_synthetic_robustness()
        
        elif data_type == 'folktables':
            if exp_type == 'single_run':
                return self.run_folktables_single()
        
        self.log(f"\n{'='*70}")
        self.log("✓ EXPERIMENT COMPLETE!")
        self.log(f"{'='*70}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/experiment_runner.py <config.yaml>")
        print("Example: python src/experiment_runner.py configs/synthetic.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    runner = ExperimentRunner(config_file)
    runner.run()
