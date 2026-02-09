"""
Experiment utilities - modular helper functions for metrics and evaluation.
Removes code duplication across experiment types.
"""

import numpy as np
import mlflow
from typing import Dict, List, Tuple


def compute_method_metrics(log_fairness: Dict[str, List[float]], 
                          log_accuracy: Dict[str, List[float]],
                          log_precision: Dict[str, List[float]] = None,
                          log_recall: Dict[str, List[float]] = None,
                          log_f1: Dict[str, List[float]] = None,
                          method_names: List[str] = None) -> Dict[str, float]:
    """Compute aggregated metrics for all methods.
    
    Centralizes metric computation - used in single runs, robustness, folktables.
    
    Args:
        log_fairness: Dict[method_name] -> list of fairness values
        log_accuracy: Dict[method_name] -> list of accuracy values
        log_precision: Dict[method_name] -> list of precision values (optional)
        log_recall: Dict[method_name] -> list of recall values (optional)
        log_f1: Dict[method_name] -> list of F1 values (optional)
        method_names: Optional subset of methods to compute metrics for
    
    Returns:
        Dict with keys: {method}_{mean,std}_{dp,acc,prec,rec,f1}
    """
    if method_names is None:
        method_names = list(log_fairness.keys())
    
    metrics = {}
    for method in method_names:
        if method not in log_fairness or method not in log_accuracy:
            continue
        
        fairness_values = np.array(log_fairness[method])
        accuracy_values = np.array(log_accuracy[method])
        
        metrics[f'{method}_mean_dp'] = float(np.mean(np.abs(fairness_values)))
        metrics[f'{method}_std_dp'] = float(np.std(fairness_values))
        metrics[f'{method}_mean_acc'] = float(np.mean(accuracy_values))
        metrics[f'{method}_std_acc'] = float(np.std(accuracy_values))
        
        # Optional performance metrics
        if log_precision and method in log_precision:
            prec_values = np.array(log_precision[method])
            metrics[f'{method}_mean_prec'] = float(np.mean(prec_values))
            metrics[f'{method}_std_prec'] = float(np.std(prec_values))
        
        if log_recall and method in log_recall:
            rec_values = np.array(log_recall[method])
            metrics[f'{method}_mean_rec'] = float(np.mean(rec_values))
            metrics[f'{method}_std_rec'] = float(np.std(rec_values))
        
        if log_f1 and method in log_f1:
            f1_values = np.array(log_f1[method])
            metrics[f'{method}_mean_f1'] = float(np.mean(f1_values))
            metrics[f'{method}_std_f1'] = float(np.std(f1_values))
    
    return metrics


def log_metrics_to_mlflow(metrics: Dict[str, float]):
    """Log metrics dictionary to MLflow.
    
    Centralizes MLflow logging pattern.
    """
    mlflow.log_metrics(metrics)


def log_experiment_results(log_fairness: Dict[str, List[float]],
                          log_accuracy: Dict[str, List[float]],
                          log_precision: Dict[str, List[float]] = None,
                          log_recall: Dict[str, List[float]] = None,
                          log_f1: Dict[str, List[float]] = None,
                          method_names: List[str] = None):
    """Compute and log all experiment metrics to MLflow."""
    metrics = compute_method_metrics(
        log_fairness, log_accuracy, log_precision, log_recall, log_f1, method_names
    )
    log_metrics_to_mlflow(metrics)
    return metrics


def compare_methods(log_fairness: Dict[str, List[float]],
                   log_accuracy: Dict[str, List[float]],
                   baseline_method: str = 'base',
                   test_method: str = 'pid') -> Tuple[float, float]:
    """Compute statistical comparison between two methods.
    
    Args:
        log_fairness, log_accuracy: Experiment logs
        baseline_method: Reference method (e.g., 'base')
        test_method: Method to compare against (e.g., 'pid')
    
    Returns:
        (mean_improvement, std_improvement) in fairness metric
    """
    from scipy import stats
    
    baseline_fairness = np.abs(np.array(log_fairness[baseline_method]))
    test_fairness = np.abs(np.array(log_fairness[test_method]))
    
    t_stat, p_value = stats.ttest_ind(baseline_fairness, test_fairness)
    
    mean_baseline = np.mean(baseline_fairness)
    mean_test = np.mean(test_fairness)
    improvement = mean_baseline - mean_test
    
    return improvement, p_value


def aggregate_results_multiple_seeds(results_list: List[Dict[str, List]]) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across multiple random seeds.
    
    Args:
        results_list: List of (log_fairness, log_accuracy, log_control) tuples
    
    Returns:
        Dict[method] -> (mean, std) for fairness metric
    """
    all_methods = results_list[0][0].keys()
    aggregated = {}
    
    for method in all_methods:
        method_fairness = []
        for log_fairness, _, _ in results_list:
            mean_fairness = np.mean(np.abs(np.array(log_fairness[method])))
            method_fairness.append(mean_fairness)
        
        aggregated[method] = (np.mean(method_fairness), np.std(method_fairness))
    
    return aggregated


def print_experiment_summary(log_fairness: Dict[str, List[float]],
                            log_accuracy: Dict[str, List[float]],
                            title: str = "Experiment Results"):
    """Pretty-print experiment metrics summary."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    
    for method in log_fairness.keys():
        fair_values = np.abs(np.array(log_fairness[method]))
        acc_values = np.array(log_accuracy[method])
        
        print(f"\n  {method.upper()}:")
        print(f"    Fairness (DP): {np.mean(fair_values):.4f} ± {np.std(fair_values):.4f}")
        print(f"    Accuracy:      {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
    
    print(f"{'='*70}\n")
