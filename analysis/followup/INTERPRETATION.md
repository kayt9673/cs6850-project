# Reading the follow-up research plots

These figures analyze the completed **`output/2` follow-up**, with 201,600 evaluations and 144 parameter settings per method. The run used seed 43, activity floor 0.15, and beta 0.5/0.7/1.0. It recorded about **2.91 hours** of planning and evaluation time. Every figure is available as PNG and PDF, with supporting CSV tables in this folder.

## What to look at

| Figure | How to read it |
| --- | --- |
| [1. Outcomes](01_outcomes.png) | Lower bars mean fewer infections; the dashed line is the reconstructed untreated baseline. Panels use different vertical scales. Means give equal weight to the activity, beta, and budget settings within each network. |
| [2. Dynamic CIB versus MIG](02_dynamic_cib_vs_mig.png) | Each cell is a matched comparison at one fixed setting. Blue/negative means CIB prevents more infections; red/positive favors MIG. Units are percentage points, not relative percent change. Near-zero labels can be rounding; consult the CSV. |
| [3. Static hybrid versus AR](03_static_hybrid_vs_ar.png) | The same convention, comparing the alpha=0.5 hybrid against static AR. Effects vary across networks and activity distributions. |
| [4. Hybrid weight](04_alpha_sensitivity.png) | Alpha=0 is pure CIB and alpha=1 is AR. An interior weight beating both endpoints would support a benefit from mixing the scores. Lines average activities and budgets at fixed beta. |
| [5. Resource use](05_resource_tradeoff.png) | Each point represents a budget cap. Read infections together with the fraction actually immunized. Static methods spend the cap; dynamic methods often stop early, so infection rates alone do not establish which mode is better. |
| [6. Cascade depth](06_cascade_depth.png) | Top: untreated outbreak depth. Bottom: dynamic outcomes within those same strata. “2+ waves” means at least two successive transmission waves; the terminal empty simulation step is excluded. Counts are unique outbreak scenarios, each evaluated at four budget caps. |
| [7. Compute cost](07_runtime.png) | Logarithmic axis; compare both effectiveness and cost. Static planning is amortized over its 20 evaluations per plan, rather than counted 20 times. This is not the latency of one planning decision. |
| [8. Paired outbreaks](08_outbreak_pairs.png) | Exact infection counts for matched outbreaks at a 10% budget cap. Below the diagonal favors the vertical-axis method; bubble size and color count overlaps. |
| [9. Win/tie/loss](09_outbreak_win_tie_loss.png) | Frequencies of better, equal, or worse final infection counts, split by untreated depth. |
| [10. Individual trials](10_individual_outbreaks.png) | All 100 trials in a preset setting, sorted by untreated size. The horizontal axis indexes separate outbreaks, not time. |

[PER_OUTBREAK.md](PER_OUTBREAK.md) explains these granular figures and the traceable per-outbreak CSV tables.

## Initial findings

Across the evenly weighted grid, static AR averages **2.23%** infected, pure CIB **1.65%**, and the alpha=0.5 hybrid **1.70%**. Dynamic AR averages **3.38%**, adapted MIG **2.29%**, pure CIB **2.32%**, and the hybrid **2.38%**. These descriptive means support CIB over AR, but do not show an overall advantage over MIG or a consistent advantage from mixing AR with CIB. Figure 2 reveals exceptions in both directions.

Of 3,600 unique untreated scenarios, **2,507 (69.6%)** have no spread, **475 (13.2%)** have one wave, and **618 (17.2%)** have two or more waves. The depth plot does not reveal a consistent CIB advantage even in the deeper group. For football's deeper outbreaks, MIG averages **4.53 infected nodes** versus **5.01 for CIB**; this pools activities, betas and budgets and does not establish why they differ. NetScience has only **15** deeper scenarios, so its subgroup is especially uncertain. MIG's lower compute cost also matters when its infection outcome is comparable.

## Limits and reproducibility

Treat these as exploratory analyses. `paired_comparisons.csv` contains mean paired differences, win/tie/loss counts and approximate 95% t intervals **per fixed setting**: five plan-level means for static comparisons and 100 paired trials for dynamic comparisons. Intervals are not corrected for the many comparisons. The plots and conditional subgroup means do not display inferential intervals; visible differences alone are not significance claims. Budgets reuse outbreak seeds and static plans reuse 20 outbreaks, so pooled rows are not all independent.

Untreated cascades were replayed from the saved activity snapshots and evaluation seeds, after checking input and relevant source hashes. No immunization planning was rerun. The original sweep differs in activity floor, beta grid and seed, so changes between sweeps cannot be attributed to increased spread alone.

Regenerate from the repository root:

```bash
python -m experiment.research_analysis output/2 --output analysis/followup
```
