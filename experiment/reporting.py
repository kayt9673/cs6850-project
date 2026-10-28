"""Regenerate summaries and plots solely from raw CSV results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = ['network', 'activity', 'beta', 'budget_fraction', 'budget', 'mode',
          'strategy', 'alpha', 'centrality', 'mc_samples', 'seed_count',
          'seed_distribution', 'per_step', 'candidate_policy']


def aggregate(raw):
    result = raw.groupby(GROUPS, dropna=False).agg(
        trials=('infection_fraction', 'count'), mean=('infection_fraction', 'mean'),
        std=('infection_fraction', 'std'), mean_immunized=('immunized', 'mean'),
        mean_utilization=('utilization', 'mean'), mean_runtime=('runtime_seconds', 'mean')
    ).reset_index()
    if 'plan_id' in raw:
        plan_means = raw.groupby(GROUPS + ['plan_id'], dropna=False)['infection_fraction'].mean().reset_index()
        plan_stats = plan_means.groupby(GROUPS, dropna=False).agg(
            plans=('infection_fraction', 'count'),
            plan_std=('infection_fraction', 'std')
        ).reset_index()
        result = result.merge(plan_stats, on=GROUPS)
    else:
        result['plans'] = result['trials']
        result['plan_std'] = result['std']
    # A static plan reused for 20 outbreaks contributes one independent plan
    # mean, not 20 independent observations of planning variability.
    def t_critical(plans):
        if plans < 2:
            return np.nan
        small = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
        if plans in small:
            return small[plans]
        df = plans - 1
        z = 1.959963984540054
        return (z + (z**3 + z)/(4*df) +
                (5*z**5 + 16*z**3 + 3*z)/(96*df**2) +
                (3*z**7 + 19*z**5 + 17*z**3 - 15*z)/(384*df**3))
    result['ci95_halfwidth'] = result['plans'].map(t_critical) * result['plan_std'] / np.sqrt(result['plans'])
    result['ci95_low'] = (result['mean'] - result['ci95_halfwidth']).clip(lower=0)
    result['ci95_high'] = (result['mean'] + result['ci95_halfwidth']).clip(upper=1)
    return result


def report(raw_path, output, plot_mode='all'):
    if plot_mode not in ('all', 'selected'):
        raise ValueError('plot_mode must be all or selected')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(pd.read_csv(raw_path))
    summary.to_csv(output / 'aggregated.csv', index=False)
    if plot_mode == 'selected':
        # Keep the complete summary CSV; only reduce rendered scenarios.
        activity = ('exponential' if 'exponential' in set(summary.activity)
                    else summary.activity.iloc[0])
        beta = min(summary.beta.unique(), key=lambda value: abs(value - .5))
        budget = min(summary.budget_fraction.unique(), key=lambda value: abs(value - .1))
        weights = summary.alpha.dropna().unique()
        alpha = min(weights, key=lambda value: abs(value - .5)) if len(weights) else None
    # Each figure fixes every other experimental parameter; never pool beta/alpha/budget.
    for x, y, title in [('budget_fraction', 'mean', 'Infection fraction versus budget'),
                         ('alpha', 'mean', 'Infection fraction versus hybrid weight'),
                         ('beta', 'mean', 'Infection fraction versus transmission'),
                         ('mode', 'mean', 'Static versus dynamic infection fraction'),
                         ('mode', 'mean_immunized', 'Mean immunizations used')]:
        data = summary[summary.strategy == 'hybrid'] if x == 'alpha' else summary
        if plot_mode == 'selected':
            data = data[data.activity == activity]
            if x != 'beta':
                data = data[data.beta == beta]
            if x != 'budget_fraction':
                data = data[data.budget_fraction == budget]
            if x != 'alpha':
                data = data[data.alpha.isna() | (data.alpha == alpha)]
        fixed = [c for c in GROUPS if c not in (x, 'strategy', 'mode', 'budget')]
        for number, (values, frame) in enumerate(data.groupby(fixed, dropna=False)):
            fig, ax = plt.subplots(figsize=(8, 5))
            for (strategy, mode), line in frame.groupby(['strategy', 'mode']):
                line = line.sort_values(x)
                label = f'{mode} {strategy}'
                if x == 'mode':
                    ax.scatter(line[x], line[y], label=label)
                else:
                    errors = line['ci95_halfwidth'].fillna(0) if y == 'mean' else None
                    ax.errorbar(line[x], line[y], yerr=errors, marker='o', label=label,
                                capsize=2)
            ax.set(xlabel=x.replace('_', ' '), ylabel=('Final infection fraction' if y == 'mean'
                   else 'Computers immunized'), title=title)
            caption = ', '.join(f'{k}={v}' for k, v in zip(fixed, values))
            fig.text(.02, .01, caption, fontsize=6, wrap=True)
            ax.legend(fontsize=7)
            fig.tight_layout(rect=(0, .08, 1, 1))
            fig.savefig(output / f'{x}-{y}-{number}.png', dpi=150)
            plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plot-mode', choices=('all', 'selected'), default='all')
    args = parser.parse_args()
    report(args.raw, args.output, plot_mode=args.plot_mode)
