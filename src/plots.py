"""
Plotting utilities for experimental results with professional styling.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def _format_label(method_name):
    """Format method name for display (handle special cases like pid -> PID)."""
    label = method_name.replace('_', ' ').title()
    # Special cases
    if label.lower() == 'pid':
        return 'PID'
    return label


def _add_year_annotations(ax, year_labels, y_position=0.95, fontsize=10):
    """Add year annotations to a plot at specified y position."""
    if not year_labels:
        return
    
    y_min, y_max = ax.get_ylim()
    y_top = y_min + (y_max - y_min) * y_position
    
    for t, year in year_labels:
        ax.text(t, y_top, str(year), 
                fontsize=fontsize, ha='center', va='top',
                color='darkgray', alpha=0.8)


def plot_single_metric(data_dict, metric_title, ylabel, output_path, all_years=None):
    """
    Plot a single metric.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping method names to lists of values.
    metric_title : str
        Title for the plot.
    ylabel : str
        Label for y-axis.
    output_path : str
        Path where to save the figure.
    all_years : list of tuples, optional
        List of (timestep, year) pairs for temporal annotation.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    colors = sns.color_palette("husl", n_colors=len(data_dict))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for idx, (method, values) in enumerate(data_dict.items()):
        label = _format_label(method)
        ax.plot(values, label=label, linewidth=2.5, color=colors[idx], alpha=0.85)
    
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(metric_title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    if all_years:
        _add_year_annotations(ax, all_years, y_position=0.95, fontsize=10)
    
    plt.tight_layout(pad=1.0)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def plot_results(logs, output_path='results.png', all_years=None):
    """
    Plot experimental results as separate images (one metric per image).
    
    Parameters:
    -----------
    logs : dict
        Dictionary with keys 'fairness', 'accuracy', 'control'
        Each value is a dict mapping method names to lists of values.
    output_path : str
        Base path where to save figures (will create _fairness, _accuracy, _control).
    all_years : list of tuples, optional
        List of (timestep, year) pairs for temporal annotation.
    """
    output_path = Path(output_path)
    base_dir = output_path.parent
    base_name = output_path.stem
    
    # Plot 1: Fairness (Demographic Parity)
    if 'fairness' in logs and logs['fairness']:
        fairness_path = base_dir / f"{base_name}_fairness.png"
        plot_single_metric(
            logs['fairness'],
            'Fairness Metric (Demographic Parity)',
            'Demographic Parity',
            str(fairness_path),
            all_years=all_years
        )
    
    # Plot 2: Accuracy
    if 'accuracy' in logs and logs['accuracy']:
        accuracy_path = base_dir / f"{base_name}_accuracy.png"
        plot_single_metric(
            logs['accuracy'],
            'Classification Accuracy',
            'Accuracy',
            str(accuracy_path),
            all_years=all_years
        )
    
    # Plot 3: Control Signal (if available) or F1
    if 'control' in logs and logs['control']:
        control_path = base_dir / f"{base_name}_control.png"
        plot_single_metric(
            logs['control'],
            'PID Control Signal',
            'Control Signal',
            str(control_path),
            all_years=all_years
        )
    elif 'f1' in logs and logs['f1']:
        f1_path = base_dir / f"{base_name}_f1.png"
        plot_single_metric(
            logs['f1'],
            'F1 Score',
            'F1 Score',
            str(f1_path),
            all_years=all_years
        )


def plot_combined_metrics(logs, output_path='combined_metrics.png', all_years=None):
    """
    Alternative layout: All performance metrics in 2x2 subplots.
    
    Parameters:
    -----------
    logs : dict
        Dictionary with fairness, accuracy, precision, recall, f1 metrics.
    output_path : str
        Path where to save the figure.
    all_years : list of tuples, optional
        List of (timestep, year) pairs for temporal annotation.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    
    colors = sns.color_palette("husl", n_colors=len(logs.get('fairness', {})))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    metrics_config = [
        ('fairness', 'Demographic Parity', axes[0, 0]),
        ('accuracy', 'Accuracy', axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('recall', 'Recall', axes[1, 1])
    ]
    
    for metric_key, metric_title, ax in metrics_config:
        if metric_key not in logs or not logs[metric_key]:
            continue
        
        for idx, (method, values) in enumerate(logs[metric_key].items()):
            label = _format_label(method)
            ax.plot(values, label=label, linewidth=2.5, color=colors[idx], alpha=0.85)
        
        ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_title} Over Time', fontsize=15, fontweight='bold', pad=12)
        ax.legend(fontsize=10, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        if all_years:
            _add_year_annotations(ax, all_years, y_position=0.95, fontsize=8)
    
    plt.tight_layout(pad=2.0)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved combined metrics plot to {output_path}")
    plt.close()


