
"""Li et al.'s AR ranking and the proposed AR + CIB hybrid ranking."""
import numpy as np
import networkx as nx


def calculate_ar(graph, activities, centrality='degree'):
    """AR(v) = C(v)^2 * -ln(a(v)/sum(a)); natural log throughout.

    Clip the activity ratio at float64 epsilon; all-zero activity gives zero.
    Max normalization in calculate_hybrid cancels any log-base scale factor.
    """
    if not graph:
        return {}
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

    if any(not np.isfinite(activities[v]) or not 0 <= activities[v] <= 1 for v in graph):
        raise ValueError('Activities must be finite and in [0, 1]')
    total = sum(activities[v] for v in graph)

    if total <= 0:
        return {v: 0.0 for v in graph}

    # Li et al.'s formula 
    return {
        v: float(
            c[v] ** 2 * (-np.log(max(activities[v] / total, np.finfo(float).eps)))
        )
        for v in graph.nodes()
    }


def normalize_scores(scores):
    maximum = max(scores.values(), default=0)

    return {
        v: (float(x / maximum) if maximum > 0 else 0.0)
        for v, x in scores.items()
    }


def calculate_hybrid(ar, cib_scores, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')

    if ar.keys() != cib_scores.keys():
        raise ValueError(
            'Both score dictionaries must have identical nodes'
        )

    a = normalize_scores(ar)
    c = normalize_scores(cib_scores)

    return {
        v: alpha * a[v] + (1 - alpha) * c[v]
        for v in ar
    }
