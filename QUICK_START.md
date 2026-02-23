# Quick Start - Running Experiments

This project uses a **unified experiment runner** for all experiment types (synthetic and real-world data).

## Usage

```bash
python3 -m src.experiment_runner <config_file>
```

## Available Experiments

### 1. Synthetic Data - Single Run
```bash
python3 -m src.experiment_runner configs/synthetic_single.yaml
```
- Quick test with single seed
- Data: Synthetically generated with drift at t=15
- Output: `results/synthetic/single_run/`

### 2. Synthetic Data - Robustness Analysis (Monte Carlo)
```bash
python3 -m src.experiment_runner configs/synthetic.yaml
```
- Tests: base, static GBR, sliding window, PID controller
- N=20 Monte Carlo seeds with statistical tests (t-tests)
- Output: `results/synthetic/robustness/`

### 3. Folktables - Single State Temporal
```bash
python3 -m src.experiment_runner configs/folktables_single.yaml
```
- Tests: Baseline vs PID controller
- Data: California only, 2014-2018
- Output: `results/folktables/single_run/`

### 4. Folktables - Multi-State Temporal
```bash
python3 -m src.experiment_runner configs/folktables.yaml
```
- Data: CA, TX, NY, 2014-2018
- Shows year transitions in plots
- Output: `results/folktables/single_run/`

### 5. Folktables - Geographic Diversity
```bash
python3 -m src.experiment_runner configs/folktables_geographic.yaml
```
- Tests: 10 US states (CA, TX, NY, FL, PA, IL, OH, MI, NC, GA)
- Data: 2018 only, geographic robustness
- Output: `results/folktables/robustness/`

### 6. Folktables - Bias Mitigation Benchmark (Static)
```bash
python3 -m src.benchmark_runner configs/folktables_benchmark.yaml
```
- Methods: baseline, reweighing, equalized odds, fairness constraint
- Data: CA, 2018 (static split)
- Output: `results/folktables/benchmark/`

### 7. Folktables - Bias Mitigation Benchmark (Temporal)
```bash
python3 -m src.benchmark_runner configs/folktables_benchmark_temporal.yaml
```
- Methods: baseline, reweighing, equalized odds, fairness constraint
- Data: CA, train 2014-2015, val 2016, test 2017-2018
- Output: `results/folktables/benchmark_temporal/`

## Output Structure

```
results/
├── synthetic/
│   ├── single_run/
│   │   ├── fig_fairness.png
│   │   ├── fig_accuracy.png
│   │   └── fig_control.png
│   ├── robustness/
│   │   ├── fig_fairness.png       (mean ± std across 20 seeds)
│   │   ├── fig_accuracy.png
│   │   ├── config_*.yaml
│   │   └── experiment_*.log
│   └── mlruns/                     (MLflow tracking database)
│       ├── 0/                      (default experiment)
│       └── <experiment_id>/
│           ├── <run_uuid>/
│           │   ├── metrics/        (18+ metrics logged)
│           │   │   ├── pid_mean_dp
│           │   │   ├── pid_vs_base_ttest_pvalue
│           │   │   └── ... 
│           │   ├── tags/           (metadata: user, git commit, run name)
│           │   └── meta.yaml       (run configuration)
│           └── ...
└── folktables/
    ├── single_run/
    │   ├── folktables_income_temporal_fairness.png
    │   ├── folktables_income_temporal_accuracy.png
    │   └── folktables_income_temporal_control.png
    └── mlruns/                     (MLflow tracking for Folktables)
```

## MLflow Experiment Tracking

All experiments are automatically tracked with **MLflow** for reproducibility and comparison.

### Logged Metrics (per run):
- **Fairness**: `{method}_mean_dp`, `{method}_std_dp` (demographic parity)
- **Accuracy**: `{method}_mean_acc`, `{method}_std_acc`
- **Statistical Tests**: `pid_vs_base_ttest_pvalue`, `pid_vs_base_ttest_statistic`

### View MLflow UI:
```bash
# For synthetic experiments
mlflow ui --backend-store-uri results/synthetic/mlruns

# For Folktables experiments  
mlflow ui --backend-store-uri results/folktables/mlruns
```

Then open **http://localhost:5000** in your browser.

### What you can do in MLflow UI:
- ✅ Compare multiple experiment runs side-by-side
- ✅ Sort/filter runs by metrics (e.g., lowest p-value)
- ✅ View experiment parameters and configuration
- ✅ Track git commit for each run
- ✅ Export comparison tables for papers
- ✅ Download metrics as CSV

### Example metrics file:
```
results/synthetic/mlruns/.../metrics/pid_vs_base_ttest_pvalue
```
Content:
```
timestamp value step
1770645033424 8.9e-94 0
```
Format: `timestamp` (ms), `value`, `step` (always 0 for summary metrics)

## Configuration Files

Edit `configs/synthetic.yaml` or `configs/folktables.yaml` to customize:
- Number of time steps
- PID parameters (Kp, Ki, Kd)
- Data settings (batch size, years, states, etc.)

## Legacy Scripts

Old scripts are in `legacy/` folder for reference:
- `legacy/run.py` - Single run (synthetic)
- `legacy/batch_runner.py` - Robustness (synthetic)
- `legacy/batch_runner_folktables.py` - Robustness (Folktables)

These are **not maintained** - use `experiment_runner.py` instead.

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
pyyaml
folktables
mlflow          # NEW: Experiment tracking
scipy           # NEW: Statistical tests
```

Install with: `pip3 install -r requirements.txt`

## Scientific Rigor Features

✅ **Experiment Tracking**: MLflow logs all metrics automatically  
✅ **Statistical Tests**: t-tests (PID vs Baseline) with p-values  
✅ **Reproducibility**: Config versioning + git commit tracking  
✅ **Robustness**: Monte Carlo testing (N=20 seeds)  
✅ **Visualization**: Mean ± std bands in plots
