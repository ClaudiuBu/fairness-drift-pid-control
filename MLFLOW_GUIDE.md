# MLflow Experiment Tracking Guide

## Overview
All experiments now log metrics to MLflow for reproducibility and comparison. The tracking system organizes runs into separate experiment directories based on the config type.

## Experiment Structure

### Synthetic Data Experiments
- **Experiment Name**: `synthetic_single` or `synthetic_robustness`
- **Location**: `results/synthetic/mlruns/`
- **Single Run**: Logs baseline + PID metrics for 1 seed
- **Robustness**: Logs metrics for N=20 seeds + statistical significance tests

### Folktables Real Data Experiments
- **Experiment Name**: `folktables_single`
- **Location**: `results/folktables/mlruns/`
- **Data**: ACS Census Income 2014-2018 (temporal drift)
- **Samples**: 862K total (CA: 183K→187K→190K→192K→117K)

## Logged Metrics

### Per-Experiment Metrics
Each run logs:
- `baseline_mean_dp` / `baseline_std_dp` - Baseline fairness (mean ± std)
- `baseline_mean_acc` / `baseline_std_acc` - Baseline accuracy (mean ± std)
- `pid_mean_dp` / `pid_std_dp` - PID fairness (mean ± std)
- `pid_mean_acc` / `pid_std_acc` - PID accuracy (mean ± std)

### Robustness Runs
Additionally logs per-seed metrics:
- For each method (baseline, pid, static, sliding): `{method}_mean_dp`, `{method}_std_dp`, etc.

## Accessing Metrics

### Via Filesystem
Metrics stored as text files in:
```
results/{data_type}/mlruns/{experiment_id}/{run_id}/metrics/{metric_name}
```

Format: `{timestamp} {value} {step}`

Example:
```bash
$ cat results/synthetic/mlruns/201883985608907603/29851a63ad5a4351aeed8bd5edd618bd/metrics/pid_mean_dp
1770647222162 0.10951471601526652 0
```

### Via MLflow UI
1. Launch MLflow UI:
```bash
cd results/{data_type}
mlflow ui
```

2. Open http://localhost:5000
3. Navigate to specific experiment
4. Compare runs side-by-side
5. View metrics, plots, and configs

## Run Name Format

**Single Runs**: `{data_type}_single_{HHMMSS}`
- Example: `synthetic_single_162701`

**Robustness Runs**: `{data_type}_robustness_{HHMMSS}`
- Example: `synthetic_robustness_162743`

**Folktables Runs**: `folktables_single_{HHMMSS}`
- Example: `folktables_single_162828`

This timestamp suffix ensures each run is uniquely identifiable in MLflow UI.

## Config Versioning

Each experiment saves its config as YAML:
```bash
results/{data_type}/config_YYYYMMDD_HHMMSS.yaml
```

This includes:
- PID controller parameters (Kp, Ki, Kd, target)
- Data stream configuration (batch size, years, states)
- Experiment type (single_run, robustness, etc.)

## Comparing Experiments

### Example: PID vs Baseline (Synthetic)
```bash
cd results/synthetic
mlflow ui  # Opens at http://localhost:5000
```

In UI:
1. Click on `synthetic_robustness` experiment
2. View multiple runs (one per seed)
3. Compare metrics across runs
4. Check statistical significance (p-value logged for t-test)

### Example: Temporal Drift (Folktables)
```bash
cd results/folktables
mlflow ui
```

In UI:
1. Click on `folktables_single` experiment
2. View fairness/accuracy across 5 years
3. Verify PID maintains lower fairness drift than baseline

## Notes

- Each config type creates separate MLflow experiment
- Multiple runs of same experiment accumulate under same experiment directory
- Run names timestamped for uniqueness (prevents overwrites)
- Config saved alongside MLflow tracking for full reproducibility
- All metrics logged with timestamp for temporal analysis

## Troubleshooting

**Issue**: No metrics directory visible
```bash
# Check if mlruns was created
ls -la results/{data_type}/mlruns/

# Verify experiment ran successfully
tail -20 results/{data_type}/config_*.yaml
```

**Issue**: Can't connect to MLflow UI
```bash
# Ensure you're in correct directory
cd results/{data_type}

# Port 5000 may be in use, try alternate port
mlflow ui --port 5001
```

**Issue**: Missing metrics in specific run
```bash
# Check which runs exist
find results/{data_type}/mlruns -name "meta.yaml" -type f

# View run metadata
cat results/{data_type}/mlruns/{exp_id}/{run_id}/meta.yaml
```
