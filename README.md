# Activity-aware cascading immunization

Compare static and dynamic network immunization using Activity Ranking (AR), Cascading Immunization Benefit (CIB), and their hybrid.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Directions

After adding datasets in `configs/validation.json` or `configs/full.json`, choose a new output directory for each run:

```powershell
python run_experiments.py --config configs/validation.json --output results/validation
python run_experiments.py --config configs/full.json --output results/full
```

Outputs include raw and aggregated CSVs, plots, and activity snapshots. To regenerate reports, run:

```powershell
python reporting.py results/full/raw.csv --output results/rebuilt
```
