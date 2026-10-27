"""Reproducible dataset-only sweeps. No work runs on import."""
import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from time import perf_counter
import importlib.metadata
import numpy as np
import pandas as pd
from diffusion import CascadeModel
from immunization import sample_seeds, static_immunization, dynamic_immunization
from network import load_dataset, generate_activities
from ranking import calculate_ar


def stream_seed(master, *labels):
    payload = json.dumps([master, *labels], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def run(config, output):
    # Preflight all paths before creating output or running any simulation.
    for key in ('betas', 'budget_fractions', 'alphas'):
        if not config[key] or any(not 0 <= x <= 1 for x in config[key]):
            raise ValueError(f'{key} must be a nonempty list of values in [0, 1]')
    for key in ('mc_samples', 'trials', 'per_step'):
        value = config.get(key, 1)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('candidate_policy', 'all') not in ('all', 'frontier'):
        raise ValueError('Unknown candidate policy')
    datasets = config['datasets']
    if not datasets:
        raise ValueError('Configure at least one dataset; no datasets are bundled')
    names = [s.get('name', Path(s['path']).stem) for s in datasets]
    if len(set(names)) != len(names):
        raise ValueError('Dataset names must be unique')
    for spec in datasets:
        for key in ('path', 'nodes_path'):
            if spec.get(key) and not Path(spec[key]).is_file():
                raise FileNotFoundError(spec[key])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'config': config, 'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in
                             ('numpy', 'networkx', 'pandas', 'matplotlib')},
                'inputs': {str(s[k]): hashlib.sha256(Path(s[k]).read_bytes()).hexdigest()
                           for s in datasets for k in ('path', 'nodes_path') if s.get(k)}}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    master = config.get('seed', 42)
    rows = []
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
                           ('dynamic', 'ar', None), ('dynamic', 'local', None)]
                methods += [(mode, 'hybrid', alpha) for mode in ('static', 'dynamic')
                            for alpha in config['alphas']]
                for trial in range(config['trials']):
                    # Evaluation randomness is inaccessible to planning; all methods share it.
                    evaluation_seed = stream_seed(master, name, distribution, beta, trial, 'evaluation')
                    for mode, strategy, alpha in methods:
                        plan_seed = stream_seed(master, name, distribution, beta, fraction,
                                                trial, mode, strategy, alpha, 'planning')
                        start = perf_counter()
                        kwargs = dict(strategy=strategy, alpha=.5 if alpha is None else alpha,
                                      seed=plan_seed, ar=ar, **planning)
                        if mode == 'static':
                            protected = static_immunization(model, activities, budget, **kwargs)
                        rng = np.random.default_rng(evaluation_seed)
                        sources = sample_seeds(model, rng, planning['seed_count'],
                                               planning['seed_distribution'], activities)
                        live = model.scenario(rng)
                        if mode == 'static':
                            state = model.finish(model.initial(sources, protected), protected, live)
                        else:
                            state, protected = dynamic_immunization(model, activities, sources, live,
                                budget, per_step=config.get('per_step', 1),
                                policy=config.get('candidate_policy', 'all'), **kwargs)
                        rows.append(dict(network=name, nodes=len(graph), edges=graph.number_of_edges(),
                            activity=distribution, activity_seed=activity_seed, beta=beta,
                            budget_fraction=fraction, budget=budget, mode=mode, strategy=strategy,
                            alpha=alpha, trial=trial, seed=master, evaluation_seed=evaluation_seed,
                            planning_seed=plan_seed, infected=len(state.infected),
                            infection_fraction=len(state.infected)/len(graph),
                            immunized=len(protected), utilization=len(protected)/budget if budget else 0,
                            steps=state.steps, runtime_seconds=perf_counter()-start,
                            centrality=centrality, mc_samples=planning['samples'],
                            seed_count=planning['seed_count'], seed_distribution=planning['seed_distribution'],
                            per_step=config.get('per_step', 1), candidate_policy=config.get('candidate_policy', 'all')))
                        # Checkpoint every completed strategy; interrupted runs retain raw data.
                        pd.DataFrame(rows).to_csv(output / 'raw.csv', index=False)
    from reporting import report
    report(output / 'raw.csv', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)
