"""Reproducible dataset-only sweeps. No work runs on import."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path
from time import perf_counter
import importlib.metadata
import numpy as np
import pandas as pd
from .diffusion import CascadeModel
from .immunization import sample_seeds, static_immunization, dynamic_immunization
from .network import load_dataset, generate_activities
from .ranking import calculate_ar


def stream_seed(master, *labels):
    payload = json.dumps([master, *labels], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def create_run_directory():
    root = Path('output')
    root.mkdir(exist_ok=True)
    number = max((int(path.name) for path in root.iterdir()
                  if path.name.isdecimal()), default=0) + 1
    while True:
        destination = root / str(number)
        try:
            destination.mkdir()
        except FileExistsError:
            number += 1
        else:
            return destination


def _manifest(config, datasets):
    code = Path(__file__).resolve().parent
    return {'config': config, 'python': platform.python_version(),
            'versions': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'networkx', 'pandas', 'matplotlib')},
            'inputs': {str(s[k]): hashlib.sha256(Path(s[k]).read_bytes()).hexdigest()
                       for s in datasets for k in ('path', 'nodes_path') if s.get(k)},
            'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(code.glob('*.py'))}}


def _write_manifest(path, metadata):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _resume_rows(path, total):
    """Count durable CSV records, trimming a torn final record if needed."""
    if not path.exists():
        return 0, set(), False
    logged_plans = set()
    count = 0
    with path.open('r+b') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
        header_line = stream.readline()
        if not header_line:
            return 0, logged_plans, False
        if not header_line.endswith(b'\n'):
            stream.truncate(0)
            return 0, logged_plans, False
        fields = next(csv.reader([header_line.decode('utf-8')]))
        required = {'evaluation_index', 'mode', 'planning_seed', 'network', 'trial', 'strategy'}
        if not required <= set(fields):
            raise ValueError('Resume CSV header is incompatible with this runner')
        valid_end = stream.tell()
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                values = next(csv.reader([line.decode('utf-8')]))
            except (UnicodeDecodeError, csv.Error):
                values = []
            if not line.endswith(b'\n') or len(values) != len(fields):
                if stream.tell() != size:
                    raise ValueError('Resume CSV has a damaged record before its end')
                stream.truncate(valid_end)
                break
            row = dict(zip(fields, values))
            try:
                index = int(row['evaluation_index'])
            except ValueError as exc:
                if stream.tell() == size:
                    stream.truncate(valid_end)
                    break
                raise ValueError('Resume CSV has an invalid evaluation index') from exc
            if index != count + 1:
                if stream.tell() == size:
                    stream.truncate(valid_end)
                    break
                raise ValueError('Resume CSV has a missing or duplicate evaluation')
            if row['mode'] == 'static':
                try:
                    logged_plans.add(int(row['planning_seed']))
                except ValueError as exc:
                    if stream.tell() == size:
                        stream.truncate(valid_end)
                        break
                    raise ValueError('Resume CSV has an invalid planning seed') from exc
            count += 1
            valid_end = stream.tell()
    if count > total:
        raise ValueError(f'Resume CSV has {count} rows; expected at most {total}')
    return count, logged_plans, True


def run(config, resume=None):
    # Preflight all paths before creating output or running any simulation.
    for key in ('betas', 'budget_fractions', 'alphas'):
        if not config[key] or any(not 0 <= x <= 1 for x in config[key]):
            raise ValueError(f'{key} must be a nonempty list of values in [0, 1]')
    for key in ('mc_samples', 'trials', 'per_step'):
        value = config.get(key, 1)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    static_plans = config.get('static_plans', config['trials'])
    if (not isinstance(static_plans, int) or static_plans < 1 or
            config['trials'] % static_plans):
        raise ValueError('static_plans must be a positive divisor of trials')
    mig_threshold = config.get('mig_threshold', 0.0)
    if not isinstance(mig_threshold, (int, float)) or not math.isfinite(mig_threshold) or mig_threshold < 0:
        raise ValueError('mig_threshold must be finite and nonnegative')
    checkpoint_rows = config.get('checkpoint_rows', 100)
    if not isinstance(checkpoint_rows, int) or checkpoint_rows < 1:
        raise ValueError('checkpoint_rows must be a positive integer')
    plot_mode = config.get('plot_mode', 'all')
    if plot_mode not in ('all', 'selected'):
        raise ValueError('plot_mode must be all or selected')
    if config.get('candidate_policy', 'all') not in ('all', 'frontier'):
        raise ValueError('Unknown candidate policy')
    datasets = config['datasets']
    if not datasets:
        raise ValueError('Configure at least one dataset; no datasets are bundled')
    if not config['activities']:
        raise ValueError('Configure at least one activity distribution')
    names = [s.get('name', Path(s['path']).stem) for s in datasets]
    if len(set(names)) != len(names):
        raise ValueError('Dataset names must be unique')
    for spec in datasets:
        for key in ('path', 'nodes_path'):
            if spec.get(key) and not Path(spec[key]).is_file():
                raise FileNotFoundError(spec[key])
    total = (len(datasets) * len(config['activities']) * len(config['betas']) *
             len(config['budget_fractions']) * config['trials'] *
             (4 + 2 * len(config['alphas'])))
    metadata = _manifest(config, datasets)
    if resume is None:
        output = create_run_directory()
        _write_manifest(output / 'manifest.json', metadata)
        resume_rows, logged_plans, has_header = 0, set(), False
    else:
        output = Path(resume)
        if not output.is_dir():
            raise FileNotFoundError(f'Resume folder does not exist: {output}')
        manifest_path = output / 'manifest.json'
        if not manifest_path.exists():
            if (output / 'raw.csv').exists():
                raise ValueError('Cannot resume raw results without a manifest')
            _write_manifest(manifest_path, metadata)
        elif json.loads(manifest_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Resume manifest differs from current config, inputs, code, or environment')
        resume_rows, logged_plans, has_header = _resume_rows(output / 'raw.csv', total)
    print(f'{"Resuming" if resume is not None else "Saving results to"} {output.resolve()}', flush=True)
    master = config.get('seed', 42)
    completed = resume_rows
    started = perf_counter()
    last_progress_time = started
    last_progress_step = completed * 20 // total
    print(f'Progress: {100 * completed / total:.2f}% '
          f'({completed:,}/{total:,} evaluations)', flush=True)
    raw_stream = (output / 'raw.csv').open('a', newline='', encoding='utf-8')
    writer = None
    pending_rows = []
    row_number = 0
    try:
        for spec in datasets:
            graph = load_dataset(spec)
            name = spec.get('name', Path(spec['path']).stem)
            centrality = config.get('centrality', 'degree')
            for distribution in config['activities']:
                activity_seed = stream_seed(master, name, distribution, 'activities')
                if distribution == 'empirical':
                    activities = {v: float(graph.nodes[v]['activity']) for v in graph}
                else:
                    activities = generate_activities(graph, distribution,
                                    rng=np.random.default_rng(activity_seed),
                                    floor=config.get('activity_floor', .05))
                pd.DataFrame({'node': list(graph), 'activity': [activities[v] for v in graph]}).to_csv(
                    output / f'activities-{activity_seed}.csv', index=False)
                ar = calculate_ar(graph, activities, centrality)
                for beta, fraction in itertools.product(config['betas'], config['budget_fractions']):
                    model = CascadeModel(graph, activities, beta)
                    budget = min(len(graph), int(math.floor(fraction * len(graph) + .5)))
                    planning = dict(samples=config['mc_samples'], seed_count=config.get('seed_count', 1),
                                    seed_distribution=config.get('seed_distribution', 'uniform'))
                    methods = [('static', 'degree', None), ('static', 'ar', None),
                               ('dynamic', 'ar', None), ('dynamic', 'mig', None)]
                    methods += [(mode, 'hybrid', alpha) for mode in ('static', 'dynamic')
                                for alpha in config['alphas']]
                    static_cache = {}
                    for trial in range(config['trials']):
                        # Evaluation randomness is inaccessible to planning; all methods share it.
                        evaluation_seed = stream_seed(master, name, distribution, beta, trial, 'evaluation')
                        for mode, strategy, alpha in methods:
                            row_number += 1
                            if row_number <= resume_rows:
                                continue
                            plan_id = trial % static_plans if mode == 'static' else trial
                            plan_seed = stream_seed(master, name, distribution, beta, fraction,
                                                    plan_id, mode, strategy, alpha, 'planning')
                            kwargs = dict(strategy=strategy, alpha=.5 if alpha is None else alpha,
                                          seed=plan_seed, ar=ar, **planning)
                            if mode == 'static':
                                key = (strategy, alpha, plan_id)
                                new_plan = key not in static_cache
                                if key not in static_cache:
                                    plan_start = perf_counter()
                                    selected = static_immunization(model, activities, budget, **kwargs)
                                    static_cache[key] = (selected, perf_counter() - plan_start)
                                protected, plan_cost = static_cache[key]
                                planning_seconds = plan_cost if new_plan and plan_seed not in logged_plans else 0.0
                            else:
                                planning_seconds = 0.0
                            evaluation_start = perf_counter()
                            rng = np.random.default_rng(evaluation_seed)
                            sources = sample_seeds(model, rng, planning['seed_count'],
                                                   planning['seed_distribution'], activities)
                            live = model.scenario(rng)
                            if mode == 'static':
                                state = model.finish(model.initial(sources, protected), protected, live)
                            else:
                                state, protected = dynamic_immunization(model, activities, sources, live,
                                    budget, per_step=config.get('per_step', 1),
                                    policy=config.get('candidate_policy', 'all'),
                                    mig_threshold=mig_threshold, **kwargs)
                            row = dict(evaluation_index=row_number,
                                network=name, nodes=len(graph), edges=graph.number_of_edges(),
                                activity=distribution, activity_seed=activity_seed, beta=beta,
                                budget_fraction=fraction, budget=budget, mode=mode, strategy=strategy,
                                alpha=alpha, trial=trial, plan_id=plan_id, static_plans=static_plans,
                                mig_threshold=mig_threshold, seed=master, evaluation_seed=evaluation_seed,
                                planning_seed=plan_seed, infected=len(state.infected),
                                infection_fraction=len(state.infected)/len(graph),
                                immunized=len(protected), utilization=len(protected)/budget if budget else 0,
                                steps=state.steps, runtime_seconds=perf_counter()-evaluation_start,
                                planning_seconds=planning_seconds,
                                centrality=centrality, mc_samples=planning['samples'],
                                seed_count=planning['seed_count'], seed_distribution=planning['seed_distribution'],
                                per_step=config.get('per_step', 1), candidate_policy=config.get('candidate_policy', 'all'))
                            if writer is None:
                                writer = csv.DictWriter(raw_stream, fieldnames=row.keys())
                                if not has_header:
                                    writer.writeheader()
                                    has_header = True
                            pending_rows.append(row)
                            if len(pending_rows) >= checkpoint_rows:
                                writer.writerows(pending_rows)
                                raw_stream.flush()
                                os.fsync(raw_stream.fileno())
                                pending_rows.clear()
                            completed += 1
                            now = perf_counter()
                            progress_step = completed * 20 // total  # five-percent milestones
                            if (progress_step > last_progress_step or
                                    now - last_progress_time >= 30 or completed == total):
                                print(f'Progress: {100 * completed / total:.2f}% '
                                      f'({completed:,}/{total:,} evaluations); '
                                      f'current dataset: {name}; '
                                      f'elapsed: {(now - started) / 60:.1f} min', flush=True)
                                last_progress_step = progress_step
                                last_progress_time = now
        if pending_rows:
            writer.writerows(pending_rows)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
    finally:
        raw_stream.close()
    print('Evaluations complete; generating aggregate results and plots...', flush=True)
    from .reporting import report
    report(output / 'raw.csv', output, plot_mode=plot_mode)
    print(f'Run complete. Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', metavar='OUTPUT_DIR',
                        help='Continue an interrupted run in this output folder')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), resume=args.resume)
