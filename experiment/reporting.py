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
    result['ci95_halfwidth'] = 1.96 * result['std'] / np.sqrt(result['trials'])
    result['ci95_low'] = (result['mean'] - result['ci95_halfwidth']).clip(lower=0)
    result['ci95_high'] = (result['mean'] + result['ci95_halfwidth']).clip(upper=1)
    return result


def report(raw_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(pd.read_csv(raw_path))
    summary.to_csv(output / 'aggregated.csv', index=False)
    # Each figure fixes every other experimental parameter; never pool beta/alpha/budget.
    for x, y, title in [('budget_fraction', 'mean', 'Infection fraction versus budget'),
                         ('alpha', 'mean', 'Infection fraction versus hybrid weight'),
                         ('beta', 'mean', 'Infection fraction versus transmission'),
                         ('mode', 'mean', 'Static versus dynamic infection fraction'),
                         ('mode', 'mean_immunized', 'Mean immunizations used')]:
        data = summary[summary.strategy == 'hybrid'] if x == 'alpha' else summary
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
    args = parser.parse_args()
    report(args.raw, args.output)
