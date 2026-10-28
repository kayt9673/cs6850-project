# Full experiment changes and research interpretation

This document summarizes the current `configs/full.json` design and the changes made to the experiment runner. It is a guide for revising the research paper, not a report of completed full-sweep results. The full sweep has not been run or timed end to end.

## Research question and common model

The experiment asks whether cascading immunization benefit (CIB), alone or combined with activity ranking (AR), reduces final infections under a fixed immunization budget. All methods use the same activity-dependent independent cascade model, with transmission probability `p_uv = beta * a_u * a_v`. The full sweep retains four networks, three synthetic activity distributions, five transmission rates, four budget fractions, five hybrid weights, 300 Monte Carlo planning samples, and 100 outbreak evaluations per setting. Each method shares an evaluation seed with its comparators for a given trial.

## Dynamic MIG baseline

The previous `dynamic/local` rule ranked a node by its immediate chance of being infected. It has been replaced by `dynamic/mig`, which uses the neighbor-risk reduction in Shang et al.'s Modified Immunization Gain (MIG), Eqs. 6–8. The score estimates how much immunizing a susceptible candidate would lower infection risk among its susceptible neighbors. Candidates are restricted to susceptible neighbors of the current infectious frontier. The full config uses `mig_threshold = 0`, selects the highest positive score, and allows at most one immunization per step and at most the common budget.

**Interpretation:** This is a *budget-constrained MIG adaptation*, not Shang et al.'s complete NICT method. Their paper uses community/similarity-based propagation probabilities and threshold selection without this fixed budget. Here MIG uses the common activity-dependent cascade probabilities so differences reflect the immunization rule under the same outbreak model. The paper draft should call the comparator an adapted MIG baseline, not Shang et al.'s original method. Cite [Shang et al. (2022)](https://doi.org/10.1109/TCYB.2020.2989427) when describing the source of the MIG score.

## Five static planning replicates

For every network, activity distribution, transmission rate, budget, and static method, the full config computes five seeded planning replicates. Each plan is evaluated on 20 independent outbreak scenarios, giving 100 evaluations per setting. Static degree and AR are deterministic, so their five replicates select the same nodes; CIB-based hybrid plans can differ. Dynamic methods still plan anew for each of the 100 trials. `raw.csv` records `plan_id` and separate planning and evaluation seeds. Its `planning_seconds` value appears only on the row where a static plan is first computed.

**Interpretation:** For stochastic static methods, the mean reflects both planning and outbreak randomness, but there are only five independent planning replicates. The aggregate 95% interval uses the five plan-level mean outcomes and a Student-t critical value. Dynamic intervals use independent trial outcomes. The paper draft should state **five plans × 20 outbreaks per plan**, and should not describe the 100 static outcomes as 100 independent plans. Planning variability is estimated imprecisely with five plans, even though there are 100 outbreak evaluations.

The shared budget is a *maximum*. Dynamic methods may use fewer immunizations when the outbreak ends or no positive MIG gain remains. Report both final infection fraction and actual immunizations used when interpreting comparisons.

## Runtime and reporting changes

- **Exact CIB shortcut:** In a sampled live-edge cascade, blocking a node that the baseline cascade never infects has zero benefit. Skipping that counterfactual simulation retains the same CIB estimate and Monte Carlo sample count. This changes runtime, not the research quantity.
- **Batched raw output and resume:** The full config writes and synchronizes `raw.csv` every 100 completed strategy evaluations (`checkpoint_rows`). It avoids repeatedly rewriting previous results. After interruption, `--resume output/N` verifies the manifest and continues from the last durable row, replaying at most one unfinished batch of 100 evaluations. It discards a torn final CSV line, checks row sequence numbers, and rebuilds reports if their generation was interrupted. The manifest and `raw.csv` serve as the checkpoint; no separate growing log is written. Inputs, code, configuration, and software versions must match the saved run. These mechanics do not change completed experiment outcomes.
- **Selected plots:** The full config saves the complete `raw.csv` and `aggregated.csv` but renders a representative subset of plots. It uses exponential activity, the transmission rate nearest 0.5 and budget fraction nearest 0.1 whenever those parameters are held fixed; for non-alpha plots it shows the hybrid weight nearest 0.5 plus non-hybrid baselines. Plots that vary beta, budget, or alpha retain the full sweep along that axis. This changes presentation only. Use the aggregate CSV for conclusions across all settings.
- **Progress output:** The runner reports the percentage and count of completed strategy evaluations at five-percent milestones or after 30 seconds when a strategy finishes. It announces report generation separately and prints the output path at completion. These percentages measure completed evaluations, not a fraction of total runtime; NetScience evaluations are slower than smaller-network evaluations.

## Paper draft edits

1. In *Promising Future Research Questions* and *Initial Steps*, replace “compare against their original methods” with a statement that the static comparator is Li et al.'s AR ranking and the dynamic comparator is a budget-constrained adaptation of Shang et al.'s MIG under the common cascade model.
2. In *Initial Steps*, specify the five-plan nested static design, the 20 independent outbreak evaluations per plan, and plan-level confidence intervals. Describe dynamic trials separately.
3. Describe budget `B` as a maximum for dynamic methods and discuss actual immunization use alongside final infections.
4. Do not treat the selected plots as the entire sweep. Base claims about performance across networks, activity distributions, budgets, and transmission rates on the complete aggregate table.

The full run's completion time on a laptop remains unverified. The smaller validation config can check correctness and output generation; it cannot establish the full sweep's runtime.
