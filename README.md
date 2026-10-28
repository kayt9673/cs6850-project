# Activity-aware cascading immunization

Compare static and dynamic network immunization using Activity Ranking (AR), Cascading Immunization Benefit (CIB), and their hybrid.

The research question is whether estimating the expected reduction in downstream infections improves static and dynamic immunization under the same activity-dependent independent cascade model and fixed budget. The dynamic comparison includes a budget-constrained adaptation of Shang et al.'s Modified Immunization Gain (MIG).

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

The runner prints the percentage of completed strategy evaluations at every 5% milestone and at least every 30 seconds after a strategy finishes. It prints a separate message while generating aggregate results and plots, then the completed output path. A percentage measures completed evaluations, not elapsed time; larger networks can make later percentages slower.

### Moderate-spread follow-up

The completed `configs/full.json` sweep is the primary experiment. `configs/spread_followup.json` is a separate exploratory follow-up chosen after inspecting that run. It keeps the same four networks, three activity distributions, four budgets, methods, 300 CIB samples, 100 outbreak evaluations, and five static plans. It raises the activity floor from 0.05 to 0.15, focuses on transmission rates 0.5, 0.7, and 1.0, and uses seed 43. It still starts each outbreak from one infected node. This yields 201,600 strategy evaluations, 60% as many as the primary sweep.

Before another sweep, [EXISTING_RUN_SPREAD_ANALYSIS.md](EXISTING_RUN_SPREAD_ANALYSIS.md) outlines an exploratory comparison by untreated outbreak depth using the completed `output/1/` data. It needs no new immunization run.

An untreated-outbreak pilot with the follow-up settings spread beyond the starting node in 30.4% of scenarios, versus 16.4% for the primary settings. The follow-up pilot's mean was 2.89 infected nodes and its 99th percentile was about 36; the primary pilot's values were 2.14 and 32. This increases the opportunity to study downstream effects without making most outbreaks large. These pilot statistics are descriptive and the follow-up should be reported separately from the primary experiment. Since the floor, transmission grid, and random seed differ, a difference between the two runs cannot be attributed to any one of those changes alone; compare strategies within each run.

Run the follow-up from the repository root with:

```bash
python -m experiment.experiments --config configs/spread_followup.json
```

Its runtime has not been measured end to end. A high-transmission football pilot at the new floor took 13.5 seconds for one plan and outbreak, close to 12.7 seconds at the original floor; the corresponding NetScience pilot took 93 seconds versus 89 seconds. The reduced transmission grid should offset some extra cascade work, but actual runtime can vary.

If a run stops, resume its numbered output folder with the same configuration:

```bash
python -m experiment.experiments --config configs/full.json --resume output/1
```

Replace `output/1` with the folder printed by your run. The runner verifies the saved configuration, dataset hashes, code, Python version, and package versions; changing any of them requires a new run. It may replay up to one unfinished batch (100 evaluations in the full config), then appends results without duplicates. A partial final CSV line is discarded. If interruption happens during report generation, the same command rebuilds reports from the complete raw results. No separate checkpoint log is created. Resume is available for runs started with this version of the runner; it cannot repair a damaged dataset or disk.

## Study design and changes to describe in the paper

**MIG comparison.** The former `dynamic/local` strategy measured only a candidate's immediate risk of infection. It has been replaced by `dynamic/mig`. Following [Shang et al., Eqs. 6–8](https://doi.org/10.1109/TCYB.2020.2989427), MIG estimates the reduction in infection risk among each candidate's susceptible neighbors if that candidate is immunized. In the current independent cascade model, the candidate's activation probability comes from the current infectious frontier and edge transmission probabilities are `p_uv = beta * a_u * a_v`. Neighbors already infected or immunized are excluded. MIG considers susceptible neighbors of the frontier, selects the highest positive score above `mig_threshold`, and is capped by `per_step` and the shared total budget. Both supplied configs use threshold 0 and one immunization per step.

This is an **adaptation of MIG, not a replication of Shang et al.'s full NICT method**: their community/similarity transmission model and unconstrained threshold-based selection are replaced by the common activity-dependent cascade model and fixed budget. The draft should say it compares against *a budget-constrained MIG adaptation under the common cascade model*, rather than against Shang et al.'s original method. The shared model isolates the immunization rule for comparison.

**Five static plans.** `configs/full.json` sets `static_plans` to 5 and `trials` to 100. For each dataset, activity distribution, transmission rate, budget, and static method, five independently seeded planning replicates are computed. Each is evaluated on 20 independent outbreaks. Degree and AR plans are deterministic, so their five replicates select the same nodes; hybrid plans with CIB can differ. Dynamic methods still make a new decision in each of the 100 outbreaks. The validation config uses three plans and three trials. `raw.csv` records `plan_id`, `planning_seed`, `evaluation_seed`, `planning_seconds`, and `runtime_seconds`. Static planning time appears only on the row where a plan is first computed; `runtime_seconds` measures evaluation time and excludes cached static planning. The two modes use the same evaluation seed for a given trial.

The draft should describe the full static design as **five independent plans × 20 outbreaks per plan**, rather than 100 independent static plans. The static mean averages over both planning and outbreak randomness, but it has only five independent planning replicates. `aggregated.csv` reports `plans` and computes the static 95% interval from the five plan-level means using a Student-t critical value. Dynamic intervals use one independent trial per plan ID. The small number of static plans limits how precisely planning variability can be estimated; do not treat the 100 static outcomes as 100 independent plans.

In the draft's *Promising Future Research Questions* and *Initial Steps* sections, replace the claim that the experiment compares against Shang's original method with the MIG adaptation described above. In *Initial Steps*, add the five-plan design and state that confidence intervals for static methods use plan-level means. Keep the claim of a common fixed budget as a **maximum**: dynamic methods, including MIG, may use fewer immunizations when the outbreak or positive-gain opportunities end.

**CIB computation and output.** For each sampled live-edge cascade, a candidate outside its baseline infected set has exactly zero marginal benefit. The implementation skips only those zero-benefit counterfactual simulations, retaining the same CIB estimate and Monte Carlo sample count. Results are written to `raw.csv` in batches of `checkpoint_rows` (100 in the full config), so an interrupted run may replay the unfinished batch. The full config's `plot_mode` is `selected`: all raw and aggregate results are saved, while plots show representative settings. These changes do not alter the estimand. The full sweep remains computationally expensive and has not been verified to finish within 24 hours on a laptop.

The full experiment's methods, statistical interpretation, and draft edits are collected in [FULL_EXPERIMENT_CHANGES.md](FULL_EXPERIMENT_CHANGES.md).

## Results

Each experiment automatically saves `raw.csv`, `aggregated.csv`, comparison plots, activity snapshots, and `manifest.json` in its numbered output folder. To rebuild reports from saved results without rerunning experiments:

```powershell
python -m experiment.reporting output/1/raw.csv --output output/1
```

Add `--plot-mode selected` to regenerate only the representative plots used by the full configuration.
