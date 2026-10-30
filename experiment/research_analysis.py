"""Descriptive research figures and paired contrasts for a completed sweep."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from .diffusion import CascadeModel
from .immunization import sample_seeds
from .network import load_dataset


SCENARIO = ['network', 'activity', 'beta', 'trial']
CELL = ['network', 'activity', 'beta', 'budget_fraction']
METHODS = {
    'static': ['Degree', 'AR', 'CIB', 'Hybrid .5'],
    'dynamic': ['AR', 'MIG', 'CIB', 'Hybrid .5'],
}
COLORS = {'Degree': '#777777', 'AR': '#d89020', 'MIG': '#208578',
          'CIB': '#3b68aa', 'Hybrid .5': '#99559e'}
CONTRASTS = [('static', 'CIB', 'AR'), ('static', 'Hybrid .5', 'AR'),
             ('dynamic', 'CIB', 'MIG'), ('dynamic', 'Hybrid .5', 'MIG'),
             ('dynamic', 'Hybrid .5', 'AR')]


def method_name(row):
    if row.strategy != 'hybrid':
        return {'degree': 'Degree', 'ar': 'AR', 'mig': 'MIG'}[row.strategy]
    return {0: 'CIB', .5: 'Hybrid .5', 1: 'AR endpoint'}.get(row.alpha, f'Hybrid {row.alpha}')


def save(fig, out, name, caption):
    fig.text(.02, .01, caption, fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(out / f'{name}.png', dpi=180)
    fig.savefig(out / f'{name}.pdf')
    plt.close(fig)


def untreated_replay(run, manifest, raw):
    """Reuse saved activity snapshots and the exact evaluation RNG order."""
    root = Path(__file__).resolve().parents[1]
    for relative, digest in manifest['inputs'].items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed since run: {path}')
    for name in ('diffusion.py', 'immunization.py', 'network.py'):
        digest = manifest.get('code', {}).get(name)
        if digest and hashlib.sha256((root / 'experiment' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Replay requires the saved run version of {name}')
    c = manifest['config']
    rows = []
    for spec in c['datasets']:
        local = dict(spec)
        for key in ('path', 'nodes_path'):
            if local.get(key) and not Path(local[key]).is_absolute():
                local[key] = str(root / local[key])
        graph = load_dataset(local)
        name = spec.get('name', Path(spec['path']).stem)
        for activity in c['activities']:
            part = raw[(raw.network == name) & (raw.activity == activity)]
            activity_seed = int(part.activity_seed.iloc[0])
            snapshot = pd.read_csv(run / f'activities-{activity_seed}.csv',
                                   dtype={'node': str}, float_precision='round_trip')
            activities = dict(zip(snapshot.node, snapshot.activity))
            for beta in c['betas']:
                model = CascadeModel(graph, activities, beta)
                scenarios = part[part.beta == beta].drop_duplicates('trial')
                for item in scenarios.itertuples():
                    rng = np.random.default_rng(int(item.evaluation_seed))
                    seeds = sample_seeds(model, rng, c.get('seed_count', 1),
                                         c.get('seed_distribution', 'uniform'), activities)
                    state = model.finish(model.initial(seeds), (), model.scenario(rng))
                    waves = state.steps - 1  # finish includes the final empty wave
                    rows.append(dict(network=name, activity=activity, beta=beta,
                                     trial=item.trial, untreated_infected=len(state.infected),
                                     untreated_fraction=len(state.infected) / len(graph),
                                     waves=waves,
                                     stratum='No spread' if waves == 0 else
                                     'One wave' if waves == 1 else '2+ waves'))
    return pd.DataFrame(rows)


def paired_table(raw):
    rows = []
    for mode, treatment, control in CONTRASTS:
        a = raw[(raw['mode'] == mode) & (raw.method == treatment)]
        b = raw[(raw['mode'] == mode) & (raw.method == control)]
        joined = a.merge(b, on=CELL + ['trial'], suffixes=('_a', '_b'), validate='one_to_one')
        assert (joined.evaluation_seed_a == joined.evaluation_seed_b).all()
        joined['delta_pp'] = 100 * (joined.infection_fraction_a - joined.infection_fraction_b)
        for values, frame in joined.groupby(CELL):
            units = (frame.groupby('plan_id_a').delta_pp.mean() if mode == 'static'
                     else frame.delta_pp)
            n = len(units)
            # Supplied sweeps use five static plans and 100 dynamic trials.
            critical = {5: 2.776, 100: 1.984}.get(n, np.nan)
            half = critical * units.std(ddof=1) / np.sqrt(n)
            mean = frame.delta_pp.mean()
            rows.append(dict(zip(CELL, values), mode=mode, treatment=treatment,
                             control=control, mean_delta_pp=mean,
                             ci95_low_pp=mean-half, ci95_high_pp=mean+half,
                             independent_units=n, paired_outbreaks=len(frame),
                             wins=int((frame.delta_pp < 0).sum()),
                             ties=int((frame.delta_pp == 0).sum()),
                             losses=int((frame.delta_pp > 0).sum())))
    return pd.DataFrame(rows)


def outbreak_pairs(raw):
    """One traceable row per paired method comparison and budget/outbreak."""
    frames = []
    measures = ['evaluation_index', 'evaluation_seed', 'planning_seed', 'plan_id',
                'infected', 'infection_fraction', 'immunized']
    baseline = ['untreated_infected', 'waves', 'stratum']
    for mode, treatment, control in CONTRASTS:
        left = raw[(raw['mode'] == mode) & (raw.method == treatment)]
        right = raw[(raw['mode'] == mode) & (raw.method == control)]
        if left.empty or right.empty:
            continue
        joined = left[CELL + ['trial'] + baseline + measures].merge(
            right[CELL + ['trial'] + measures], on=CELL + ['trial'],
            suffixes=('_treatment', '_control'), validate='one_to_one')
        if not (joined.evaluation_seed_treatment == joined.evaluation_seed_control).all():
            raise ValueError('Paired methods must share the evaluation seed')
        joined['mode'], joined['treatment'], joined['control'] = mode, treatment, control
        joined['delta_infected'] = joined.infected_treatment - joined.infected_control
        joined['delta_immunized'] = joined.immunized_treatment - joined.immunized_control
        joined['outcome'] = np.select([joined.delta_infected < 0, joined.delta_infected > 0],
                                      ['Win', 'Loss'], default='Tie')
        frames.append(joined)
    return pd.concat(frames, ignore_index=True)


def granular_figures(raw, nets, config, out):
    pairs = outbreak_pairs(raw)
    pairs.to_csv(out / 'outbreak_pairs.csv', index=False)
    budgets = config['budget_fractions']
    budget = min(budgets, key=lambda x: abs(x - .1))
    comparisons = [('static', 'Hybrid .5', 'AR'), ('dynamic', 'CIB', 'MIG')]
    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, (mode, treatment, control) in enumerate(comparisons):
        for j, net in enumerate(nets):
            ax = axes[i, j]
            data = pairs[(pairs['mode'] == mode) & (pairs.treatment == treatment) &
                         (pairs.control == control) & (pairs.network == net) &
                         (pairs.budget_fraction == budget)]
            counts = data.groupby(['infected_control', 'infected_treatment']).size().reset_index(name='count')
            image = ax.scatter(counts.infected_control, counts.infected_treatment,
                               c=counts['count'], s=12 + 18*np.log1p(counts['count']),
                               norm=LogNorm(vmin=1, vmax=max(2, counts['count'].max())),
                               cmap='viridis', edgecolors='#555555', linewidths=.25)
            bound = max(data.infected_control.max(), data.infected_treatment.max(), 1)
            ax.plot([0, bound], [0, bound], '--', color='#bb4444', lw=1)
            ax.set(xlim=(-.4, bound+.4), ylim=(-.4, bound+.4),
                   xlabel=f'{control}: infected nodes', ylabel=f'{treatment}: infected nodes',
                   title=f'{net} / {mode} (n={len(data)})')
            fig.colorbar(image, ax=ax, label='Outbreak pairs (log count)', fraction=.045)
    fig.suptitle(f'Individual paired outbreaks at a {budget:.0%} budget cap')
    save(fig, out, '08_outbreak_pairs', 'Below diagonal = treatment better; above = control better. Bubbles show exact infection counts; size/color indicate overlapping pairs. All activities/betas; one budget avoids four repeats.')

    strata = ['No spread', 'One wave', '2+ waves']
    outcomes = ['Win', 'Tie', 'Loss']
    records = []
    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, (mode, treatment, control) in enumerate(comparisons):
        for j, net in enumerate(nets):
            data = pairs[(pairs['mode'] == mode) & (pairs.treatment == treatment) &
                         (pairs.control == control) & (pairs.network == net)]
            ax = axes[i, j]
            bottom = np.zeros(3)
            totals = [len(data[data.stratum == s]) for s in strata]
            for outcome, color in zip(outcomes, ['#208578', '#cccccc', '#c75b5b']):
                counts = [int(((data.stratum == s) & (data.outcome == outcome)).sum()) for s in strata]
                percentages = np.divide(100*np.array(counts), totals,
                                        out=np.zeros(3), where=np.array(totals)>0)
                ax.bar(range(3), percentages, bottom=bottom, color=color, label=outcome)
                bottom += percentages
                for s, n, total in zip(strata, counts, totals):
                    records.append(dict(network=net, mode=mode, treatment=treatment,
                                        control=control, stratum=s, outcome=outcome,
                                        count=n, budget_outbreak_pairs=total))
            ax.set_xticks(range(3), [f'{s}\nn={n}' for s, n in zip(strata, totals)])
            ax.set(ylim=(0, 100), ylabel='Budget/outbreak pairs (%)',
                   title=f'{net} / {mode}\n{treatment} vs {control}')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('How often does a method win, tie, or lose on the same outbreak?')
    save(fig, out, '09_outbreak_win_tie_loss', 'Win means fewer infected nodes, regardless of immunizations used. Strata use untreated depth. Counts include all four budgets and are not independent sample sizes.')
    pd.DataFrame(records).to_csv(out / 'outbreak_win_tie_loss.csv', index=False)

    # Preset the illustrative setting rather than selecting trials by who wins.
    activity = ('negative_binomial' if 'negative_binomial' in config['activities']
                else config['activities'][0])
    beta = max(config['betas'])
    case = raw[(raw.activity == activity) & (raw.beta == beta) &
               (raw.budget_fraction == budget)].copy()
    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    trace = []
    for j, net in enumerate(nets):
        data = case[case.network == net]
        order = data.drop_duplicates('trial').sort_values(['untreated_infected', 'trial'])
        trials = order.trial.tolist()
        for rank, item in enumerate(order.itertuples(), 1):
            trace.append(dict(network=net, rank=rank, trial=item.trial,
                              activity=activity, beta=beta, budget_fraction=budget,
                              evaluation_seed=item.evaluation_seed,
                              untreated_infected=item.untreated_infected))
        for i, mode in enumerate(['static', 'dynamic']):
            ax = axes[i, j]
            ax.plot(range(1, len(trials)+1), order.untreated_infected,
                    color='black', ls='--', lw=1, label='Untreated')
            names = ['AR', 'CIB', 'Hybrid .5'] if mode == 'static' else ['MIG', 'CIB', 'Hybrid .5']
            for name in names:
                series = data[(data['mode'] == mode) & (data.method == name)].set_index('trial').reindex(trials)
                ax.plot(range(1, len(trials)+1), series.infected, marker='.', ms=3,
                        lw=.8, color=COLORS[name], alpha=.8, label=name)
            ax.set(title=f'{net} / {mode}', xlabel='Outbreak rank by untreated size',
                   ylabel='Final infected nodes')
        axes[0, j].legend(fontsize=7)
        axes[1, j].legend(fontsize=7)
    fig.suptitle(f'All {config["trials"]} individual outbreaks: {activity}, beta={beta}, budget={budget:.0%}')
    save(fig, out, '10_individual_outbreaks', 'Both rows use the same trial order, sorted only by untreated size, then trial ID. Lines connect separate outbreaks, not infection timelines. See individual_outbreak_order.csv for seeds and trial IDs.')
    pd.DataFrame(trace).to_csv(out / 'individual_outbreak_order.csv', index=False)


def generate(run, out):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'manifest.json').read_text())
    c = manifest['config']
    raw = pd.read_csv(run / 'raw.csv', float_precision='round_trip',
                      dtype={key: 'uint64' for key in
                             ('activity_seed', 'evaluation_seed', 'planning_seed')})
    expected = (len(c['datasets']) * len(c['activities']) * len(c['betas']) *
                len(c['budget_fractions']) * c['trials'] * (4 + 2 * len(c['alphas'])))
    if len(raw) != expected or raw.duplicated(CELL + ['trial', 'mode', 'strategy', 'alpha']).any():
        raise ValueError('Run is incomplete or has duplicate outcomes')
    raw['method'] = raw.apply(method_name, axis=1)
    baseline = untreated_replay(run, manifest, raw)
    baseline.to_csv(out / 'untreated_scenarios.csv', index=False)
    raw = raw.merge(baseline, on=SCENARIO, validate='many_to_one')
    if (raw.infected > raw.untreated_infected).any():
        raise ValueError('Untreated replay disagrees with saved treated scenarios')
    paired = paired_table(raw)
    paired.to_csv(out / 'paired_comparisons.csv', index=False)
    nets = [s['name'] for s in c['datasets']]
    selected = raw[((raw['mode'] == 'static') & raw.method.isin(METHODS['static'])) |
                   ((raw['mode'] == 'dynamic') & raw.method.isin(METHODS['dynamic']))].copy()
    summary = selected.groupby(['network', 'mode', 'method']).agg(
        infection_fraction=('infection_fraction', 'mean'),
        infected=('infected', 'mean'), immunized=('immunized', 'mean'),
        utilization=('utilization', 'mean')).reset_index()
    summary.to_csv(out / 'method_overview.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, mode in enumerate(METHODS):
        for j, net in enumerate(nets):
            ax = axes[i, j]
            data = summary[(summary.network == net) & (summary['mode'] == mode)].set_index('method')
            names = METHODS[mode]
            ax.bar(names, 100*data.loc[names, 'infection_fraction'], color=[COLORS[x] for x in names])
            ax.axhline(100*baseline[baseline.network == net].untreated_fraction.mean(),
                       color='black', ls='--', lw=1, label='Untreated')
            ax.set_title(f'{net} / {mode}')
            ax.set_ylabel('Mean final infected (%)')
            ax.tick_params(axis='x', rotation=30)
    axes[0, 0].legend()
    fig.suptitle('Follow-up outcomes relative to untreated outbreaks')
    save(fig, out, '01_outcomes', 'Equal weight per activity, beta and budget setting. Descriptive means; no pooled confidence intervals. Compare modes alongside resource use (Figure 5).')

    for mode, treatment, control, filename in [
        ('dynamic', 'CIB', 'MIG', '02_dynamic_cib_vs_mig'),
        ('static', 'Hybrid .5', 'AR', '03_static_hybrid_vs_ar')]:
        table = paired[(paired['mode'] == mode) & (paired.treatment == treatment) & (paired.control == control)]
        limit = max(abs(table.mean_delta_pp).max(), .001)
        fig, axes = plt.subplots(len(nets), len(c['activities']), figsize=(12, 11), squeeze=False)
        for i, net in enumerate(nets):
            for j, activity in enumerate(c['activities']):
                ax = axes[i, j]
                data = table[(table.network == net) & (table.activity == activity)].pivot(
                    index='beta', columns='budget_fraction', values='mean_delta_pp').reindex(
                    index=c['betas'], columns=c['budget_fractions'])
                im = ax.imshow(data, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
                for y in range(len(data.index)):
                    for x in range(len(data.columns)):
                        value = data.iloc[y, x]
                        ax.text(x, y, f'{value:+.2f}', ha='center', va='center', fontsize=8,
                                color='white' if abs(value) > .6*limit else 'black')
                ax.set_xticks(range(len(data.columns)), [f'{v:.0%}' for v in data.columns])
                ax.set_yticks(range(len(data.index)), data.index)
                ax.set(xlabel='Budget cap', ylabel='Beta', title=f'{net} / {activity}')
        fig.suptitle(f'{mode.capitalize()}: {treatment} minus {control} (percentage points infected)')
        fig.tight_layout(rect=(0, .065, .93, .95))
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02, label='Positive = treatment worse')
        # Avoid tight_layout after the shared colorbar has reserved space.
        fig.text(.02, .015, 'Each cell uses 100 paired outbreaks. Blue = treatment better; red = worse. Intervals and win/tie/loss counts are in paired_comparisons.csv.', fontsize=8)
        fig.savefig(out / f'{filename}.png', dpi=180)
        fig.savefig(out / f'{filename}.pdf')
        plt.close(fig)

    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, mode in enumerate(METHODS):
        for j, net in enumerate(nets):
            ax = axes[i, j]
            data = raw[(raw['mode'] == mode) & (raw.network == net) & (raw.strategy == 'hybrid')]
            for beta, part in data.groupby('beta'):
                line = part.groupby('alpha').infection_fraction.mean()
                ax.plot(line.index, 100*line, marker='o', label=f'beta={beta}')
            ax.set(title=f'{net} / {mode}', xlabel='Alpha (0=CIB, 1=AR)', ylabel='Mean final infected (%)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Does mixing AR with CIB improve the outcome?')
    save(fig, out, '04_alpha_sensitivity', 'Each point averages all activities and budgets at a fixed beta. Lines are descriptive; uncertainty and networks should be examined separately.')

    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, mode in enumerate(METHODS):
        for j, net in enumerate(nets):
            ax = axes[i, j]
            data = selected[(selected['mode'] == mode) & (selected.network == net)].copy()
            data['protected_pct'] = 100*data.immunized/data.nodes
            for name in METHODS[mode]:
                line = data[data.method == name].groupby('budget_fraction').agg(
                    x=('protected_pct', 'mean'), y=('infection_fraction', 'mean'))
                ax.plot(line.x, 100*line.y, marker='o', color=COLORS[name], label=name)
            ax.set(title=f'{net} / {mode}', xlabel='Actually immunized (%)', ylabel='Mean final infected (%)')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Actual immunization use versus infections')
    save(fig, out, '05_resource_tradeoff', 'Points average activity and beta at each budget cap; connected in increasing budget order. Dynamic methods can stop early, so equal caps do not imply equal spending.')

    strata = ['No spread', 'One wave', '2+ waves']
    conditional = raw[raw['mode'] == 'dynamic'].groupby(['network', 'stratum', 'method']).agg(
        infected=('infected', 'mean'), immunized=('immunized', 'mean'),
        budget_outbreak_pairs=('trial', 'size')).reset_index()
    conditional.to_csv(out / 'conditional_dynamic.csv', index=False)
    depth = baseline.groupby(['network', 'stratum']).size().unstack(fill_value=0).reindex(columns=strata, fill_value=0)
    depth.to_csv(out / 'untreated_depth_counts.csv')
    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for j, net in enumerate(nets):
        counts = depth.loc[net]
        axes[0, j].bar(strata, 100*counts/counts.sum(), color=['#a6a6a6', '#e3b566', '#668fae'])
        axes[0, j].set(title=net, ylabel='Untreated scenarios (%)')
        axes[0, j].tick_params(axis='x', rotation=20)
        ax = axes[1, j]
        for offset, method in enumerate(['MIG', 'CIB', 'Hybrid .5']):
            values = conditional[(conditional.network == net) & (conditional.method == method)].set_index('stratum').reindex(strata)
            ax.bar(np.arange(3)+(offset-1)*.24, values.infected, width=.24, label=method, color=COLORS[method])
        ax.set_xticks(range(3), [f'{s}\nn={int(counts[s])}' for s in strata])
        ax.set_ylabel('Mean final infected (nodes)')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Does CIB help when the untreated outbreak has downstream spread?')
    save(fig, out, '06_cascade_depth', 'Strata use untreated outcomes, never treated outcomes. n counts unique scenarios; bottom panels reuse each across four budgets. Descriptive exploratory subgroups, no significance claim.')

    cost = selected.copy()
    cost['cost_ms'] = 1000*(cost.runtime_seconds + cost.planning_seconds)
    costs = cost.groupby(['network', 'mode', 'method']).cost_ms.mean().reset_index()
    costs.to_csv(out / 'amortized_runtime.csv', index=False)
    fig, axes = plt.subplots(2, len(nets), figsize=(15, 8), squeeze=False)
    for i, mode in enumerate(METHODS):
        for j, net in enumerate(nets):
            ax = axes[i, j]
            values = costs[(costs.network == net) & (costs['mode'] == mode)].set_index('method')
            names = METHODS[mode]
            ax.bar(names, values.loc[names, 'cost_ms'], color=[COLORS[x] for x in names])
            ax.set_yscale('log')
            ax.set(title=f'{net} / {mode}', ylabel='Amortized compute (ms, log scale)')
            ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Compute cost of the immunization rules')
    save(fig, out, '07_runtime', 'Includes static planning once per plan, amortized over its 20 evaluations; dynamic timing includes online planning. Excludes CSV/report overhead. Not individual decision latency.')
    granular_figures(raw, nets, c, out)
    print(f'Generated ten figure sets (PNG/PDF) and source tables in {out.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.run, args.output)
