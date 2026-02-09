# Quick Start - Running Experiments

This project uses a **unified experiment runner** for all experiment types (synthetic and real-world data).

## Usage

```bash
python3 -m src.experiment_runner <config_file>
```

## Available Experiments

### 1. Synthetic Data - Robustness Analysis
```bash
python3 -m src.experiment_runner configs/synthetic.yaml
```
- Tests: base, static GBR, sliding window, PID controller
- Data: Synthetically generated with controlled drift
- Output: `results/synthetic/robustness/`

### 2. Folktables (Real Census Data) - Single Run
```bash
python3 -m src.experiment_runner configs/folktables.yaml
```
- Tests: Baseline vs PID controller
- Data: ACS Census income prediction (2014-2018)
- Output: `results/folktables/single_run/`

## Output Structure

```
results/
├── synthetic/
│   └── robustness/
│       ├── robustness_fairness.png
│       ├── robustness_accuracy.png
│       ├── config_*.yaml          (saved configuration)
│       └── experiment_*.log       (experiment log)
└── folktables/
    └── single_run/
        ├── folktables_income_temporal_fairness.png
        ├── folktables_income_temporal_accuracy.png
        ├── folktables_income_temporal_control.png
        ├── config_*.yaml
        └── experiment_*.log
```

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
```

Install with: `pip3 install -r requirements.txt`
