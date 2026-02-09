# Folktables Integration Guide

## Overview

Acest proiect suportă acum atât **date sintetice** cât și **date reale** din **Folktables** (American Community Survey).

## Avantajele Folktables

1. **Date Reale**: Date din census cu distribuții și drift natural
2. **Multiple Task-uri**: Predicție venit, angajare, mobilitate, etc.
3. **Atribute Sensibile**: SEX, RAC1P (rasă), AGEP (vârstă)
4. **Drift Temporal**: Compararea anilor 2014-2018
5. **Diverse State-uri**: Date din toate statele US

## Utilizare Rapidă

### 1. Date Sintetice (Original)

```bash
python run.py
```

### 2. Date Reale Folktables

```bash
python run_folktables.py
```

## Exemple de Configurare

### Exemplu 1: Income Prediction cu Gender Fairness

```python
from src.data import FolktablesDataStream

stream = FolktablesDataStream(
    task='income',              # ACSIncome task
    states=['CA'],              # California
    years=[2018],
    sensitive_attribute='SEX',  # Male vs Female
    batch_size=500
)

X, y, A = stream.get_batch()
```

### Exemplu 2: Employment Prediction cu Race Fairness

```python
stream = FolktablesDataStream(
    task='employment',
    states=['NY', 'TX'],        # Multiple states
    sensitive_attribute='RAC1P', # Race (White vs Non-White)
    batch_size=1000
)
```

### Exemplu 3: Multi-Year Drift

Pentru a simula drift temporal real, poți folosi date din ani diferiți:

```python
# Year 2014 (training)
stream_2014 = FolktablesDataStream(
    task='income',
    states=['CA'],
    years=[2014],
    sensitive_attribute='SEX',
    batch_size=500
)

# Year 2018 (testing - cu drift natural)
stream_2018 = FolktablesDataStream(
    task='income',
    states=['CA'],
    years=[2018],
    sensitive_attribute='SEX',
    batch_size=500
)
```

## Task-uri Disponibile

### ACSIncome
- **Target**: Predicție dacă venitul > $50K
- **Features**: Educație, ore lucrate, ocupație, etc.
- **Utilizare**: `task='income'`

### ACSEmployment
- **Target**: Predicție dacă persoana este angajată
- **Features**: Vârstă, educație, disabilitate, etc.
- **Utilizare**: `task='employment'`

## Atribute Sensibile

### SEX
```python
sensitive_attribute='SEX'
# 1 = Male, 2 = Female -> binarizat ca 1=Male, 0=Female
```

### RAC1P (Race)
```python
sensitive_attribute='RAC1P'
# 1 = White, altele = Non-White
```

### AGEP (Age)
```python
sensitive_attribute='AGEP'
# Binarizat la median: >median = 1, <=median = 0
```

## State Codes

State-uri disponibile (exemple):
- `CA` - California
- `NY` - New York
- `TX` - Texas
- `FL` - Florida
- `PA` - Pennsylvania
- etc. (toate cele 50 de state + DC)

## Structura Fișierelor Modificate

```
src/
├── data.py              # MODIFICAT: adăugat FolktablesDataStream
├── plots.py             # MODIFICAT: suport pentru filename_prefix
├── experiment.py        # NESCHIMBAT (compatibil)
├── model.py            # NESCHIMBAT
├── fairness.py         # NESCHIMBAT
└── pid.py              # NESCHIMBAT

run.py                   # Original (date sintetice)
run_folktables.py        # NOU (date reale)
```

## Comparație Date Sintetice vs Reale

| Aspect | Date Sintetice | Folktables |
|--------|---------------|------------|
| Control Drift | ✅ Controlat exact | ⚠️ Natural, implicit |
| Reproducibilitate | ✅ Perfect | ✅ Bună (aceiași date) |
| Realitate | ❌ Simulat | ✅ Date reale |
| Dimensionalitate | 2 features | ~10 features |
| Complexitate | Simplă | Complexă |
| Interpretare | Ușoară | Mai dificilă |

## Parametri PID Recomandați

### Pentru Date Sintetice
```python
pid_kp=10.0
pid_ki=1.0
pid_kd=0.5
```

### Pentru Folktables
```python
# Mai conservativ (date mai zgomotoase)
pid_kp=5.0
pid_ki=0.5
pid_kd=0.2
```

## Tips & Tricks

### 1. Verificare Download Date

Prima rulare descarcă datele (~200MB):
```python
# Datele se salvează în ~/.folktables/
```

### 2. Debugging

Pentru a vedea dimensiunile:
```python
stream = FolktablesDataStream(task='income', states=['CA'])
print(f"Total samples: {stream.total_samples}")
print(f"Feature dim: {stream.X_all.shape[1]}")
```

### 3. Batch Size vs Dataset Size

```python
# California ACS 2018 Income: ~350K samples
# Poți rula multe batches!
batch_size = 500
T = stream.total_samples // batch_size  # ~700 time steps
```

## Troubleshooting

### Eroare: "No module named 'folktables'"
```bash
pip3 install folktables
```

### Download-ul durează prea mult
- Normal pentru prima rulare
- Datele se cached local
- Consider să folosești un singur state inițial

### Memory Issues
- Reduce numărul de state-uri
- Reduce batch_size
- Folosește un singur an

## Next Steps

### 1. Experimente Comparative
Rulează același experiment cu ambele tipuri de date:
```bash
python run.py              # Sintetic
python run_folktables.py   # Real
```

### 2. Tuning Parametri PID
Testează diferite valori Kp, Ki, Kd pe date reale.

### 3. Multiple States/Years
Creează drift explicit combinând state-uri sau ani diferiți.

### 4. Custom Drift
Modifică `FolktablesDataStream` pentru a introduce drift artificial în date reale.

## Referințe

- **Folktables Paper**: Ding et al., "Retiring Adult: New Datasets for Fair Machine Learning" (NeurIPS 2021)
- **Documentation**: https://github.com/socialfoundations/folktables
- **ACS Data**: https://www.census.gov/programs-surveys/acs
