# Activity-aware cascading immunization

Compare static and dynamic network immunization using Activity Ranking (AR), Cascading Immunization Benefit (CIB), and their hybrid.

Code is in `experiment/`, settings in `configs/`, datasets in `data/`, and tests in `tests/`.

## Datasets

| Dataset | Nodes | Edges | Local folder |
| --- | ---: | ---: | --- |
| [Karate](https://networkrepository.com/karate.php) | 34 | 78 | `data/karate/` |
| [Dolphins](https://networkrepository.com/dolphins.php) | 62 | 159 | `data/dolphins/` |
| [College football](https://networks.skewed.de/net/football) | 115 | 613 | `data/football/` |
| [NetScience](https://networks.skewed.de/net/netscience) | 1,589 | 2,742 | `data/netscience/` |

Activities are synthetic; edge weights are kept as metadata. Sources and hashes of each dataset are in `data/sources.json`.

## How to run

Run all commands from the repository root.

1. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Prepare dataset CSVs (downloads missing archives and reuses existing ones):

   ```powershell
   python -m experiment.prepare_datasets --download
   ```

   Skip this step if the prepared CSVs are already present. To rebuild without internet access, omit `--download`.

3. Run automated tests:

   ```powershell
   python -m unittest discover -s tests -v
   ```

4. Run the smaller validation configuration across all four datasets:

   ```powershell
   python -m experiment.experiments --config configs/validation.json
   ```

5. Run the full sweep when ready (substantially more expensive, especially NetScience):

   ```powershell
   python -m experiment.experiments --config configs/full.json
   ```

Both configs control datasets, activity distributions, budgets, transmission rates, hybrid weights, Monte Carlo samples, and trial counts. Each run automatically creates the next numbered folder: `output/1/`, `output/2/`, and so on. 

## Results

Each experiment automatically saves `raw.csv`, `aggregated.csv`, comparison plots, activity snapshots, and `manifest.json` in its numbered output folder. To rebuild reports from saved results without rerunning experiments:

```powershell
python -m experiment.reporting output/1/raw.csv --output output/1
```
