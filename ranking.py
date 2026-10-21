
"""Li et al.'s AR ranking and our proposed cascade-size hybrid ranking."""
import numpy as np
import networkx as nx
from diffusion import simulate_cascade


def calculate_ar(graph, activities, centrality='degree'):
    if centrality == 'degree':
        c = nx.degree_centrality(graph)

    elif centrality == 'betweenness':
        c = nx.betweenness_centrality(graph)

    elif centrality == 'k_core':
        core = nx.core_number(graph)
        c = {v: float(value) for v, value in core.items()}

    elif centrality == 'eigenvector':
        c = nx.eigenvector_centrality(graph, max_iter=2000)

    else:
        raise ValueError(f'Unknown centrality: {centrality}')

    total = sum(activities.values())

    if total <= 0:
        raise ValueError('Total activity must be positive')

    # Li et al.'s formula 
    return {
        v: float(
            c[v] * (-np.log(activities[v] / total)) ** 2
        )
        for v in graph.nodes()
    }


def estimate_cascade_sizes(
    graph, activities, beta, simulations=300, seed=42
):
    if simulations < 1:
        raise ValueError('simulations must be positive')

    rng = np.random.default_rng(seed)

    return {
        v: float(np.mean([
            len(simulate_cascade(
                graph, activities, beta, [v], rng=rng
            ))
            for _ in range(simulations)
        ]))
        for v in graph.nodes()
    }


def normalize_scores(scores):
    maximum = max(scores.values(), default=0)

    return {
        v: (float(x / maximum) if maximum > 0 else 0.0)
        for v, x in scores.items()
    }


def calculate_hybrid(ar, cascade_sizes, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')

    if ar.keys() != cascade_sizes.keys():
        raise ValueError(
            'Both score dictionaries must have identical nodes'
        )

    a = normalize_scores(ar)
    c = normalize_scores(cascade_sizes)

    return {
        v: alpha * a[v] + (1 - alpha) * c[v]
        for v in ar
    }


def select_top_k(scores, k):
    if not 0 <= k <= len(scores):
        raise ValueError(
            'k must be between 0 and number of nodes'
        )

    return set(
        sorted(
            scores,
            key=lambda v: (-scores[v], v)
        )[:k]
    )
