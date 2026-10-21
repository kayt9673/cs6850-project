"""Generates network structures and assigns node activities using
exponential, power-law, negative binomial, or mixed distributions. 
"""
import networkx as nx
import numpy as np


def generate_graph(n=100, kind='erdos_renyi', seed=42):
    if kind == 'erdos_renyi':
        graph = nx.erdos_renyi_graph(n, 0.06, seed=seed)
    elif kind == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif kind == 'watts_strogatz':
        graph = nx.connected_watts_strogatz_graph(
            n, 6, 0.15, tries=100, seed=seed
        )
    else:
        raise ValueError(f'Unknown graph type: {kind}')

    if not nx.is_connected(graph):
        graph = graph.subgraph(
            max(nx.connected_components(graph), key=len)
        ).copy()

    return graph


def generate_activities(graph, distribution='mixed', rng=None, floor=0.05):
    """Sample activities, normalize globally to [floor, 1].

    Distribution parameters are illustrative, not exact Li et al. replications.
    """
    rng = np.random.default_rng() if rng is None else rng
    nodes = list(graph.nodes())
    n = len(nodes)

    if distribution == 'exponential':
        raw = rng.exponential(scale=1.0, size=n)

    elif distribution == 'power_law':
        raw = rng.pareto(a=2.5, size=n) + 1.0

    elif distribution == 'negative_binomial':
        raw = rng.negative_binomial(
            n=5, p=0.5, size=n
        ).astype(float)

    elif distribution == 'mixed':
        groups = rng.integers(0, 3, size=n)
        raw = np.empty(n)

        for i, group in enumerate(groups):
            if group == 0:
                raw[i] = rng.exponential(scale=1.0)
            elif group == 1:
                raw[i] = rng.pareto(a=2.5) + 1.0
            else:
                raw[i] = rng.negative_binomial(n=5, p=0.5)

    else:
        raise ValueError(
            f'Unknown activity distribution: {distribution}'
        )

    if not 0 < floor <= 1:
        raise ValueError('floor must be in (0, 1]')

    maximum = max(float(raw.max()), 1e-12)
    scaled = np.maximum(raw / maximum, floor)

    return dict(zip(nodes, map(float, scaled)))
