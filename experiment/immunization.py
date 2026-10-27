"""Greedy marginal CIB and static/dynamic interventions."""
import numpy as np
from .diffusion import node_key
from .ranking import calculate_ar, calculate_hybrid


def sample_seeds(model, rng, count=1, distribution='uniform', activities=None):
    if not 0 <= count <= len(model.nodes):
        raise ValueError('Invalid seed count')
    weights = None
    if distribution == 'activity':
        weights = np.array([activities[v] for v in model.nodes], dtype=float)
        if weights.sum() <= 0 or np.count_nonzero(weights) < count:
            raise ValueError('Insufficient positive activity for seed sampling')
        weights /= weights.sum()
    elif distribution != 'uniform':
        raise ValueError('Unknown seed distribution')
    indices = rng.choice(len(model.nodes), size=count, replace=False, p=weights)
    return {model.nodes[int(i)] for i in indices}


def estimate_cib(model, candidates, immunized=(), state=None, samples=100,
                 seed=42, seed_count=1, seed_distribution='uniform', activities=None):
    if samples < 1:
        raise ValueError('samples must be positive')
    blocked = set(immunized)
    candidates = sorted(candidates, key=node_key)
    if not set(candidates) <= set(model.nodes) - blocked - (state.infected if state else set()):
        raise ValueError('CIB candidates must be susceptible and non-immunized')
    rng = np.random.default_rng(seed)
    scores = dict.fromkeys(candidates, 0.0)
    for _ in range(samples):
        live = model.scenario(rng)
        sources = (sample_seeds(model, rng, seed_count, seed_distribution, activities)
                   if state is None else None)
        initial = model.initial(sources, blocked) if state is None else state
        baseline = len(model.finish(initial, blocked, live).infected)
        for v in candidates:
            protected = blocked | {v}
            initial = model.initial(sources, protected) if state is None else state
            scores[v] += (baseline - len(model.finish(initial, protected, live).infected)) / samples
    return scores


def local_benefit(model, candidates, state):
    """Expected immediate infections avoided; deliberately not Shang's MIG."""
    probabilities = dict(zip(model.arcs, model.probabilities))
    return {v: 1 - np.prod([1 - probabilities.get((u, v), 0)
                            for u in state.frontier]) for v in candidates}


def choose(scores):
    return min(scores, key=lambda v: (-scores[v], node_key(v)))


def validate_budget(budget, per_step=1):
    if not isinstance(budget, int) or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    if not isinstance(per_step, int) or per_step < 1:
        raise ValueError('per_step must be a positive integer')


def decision(model, activities, candidates, blocked, state, strategy, ar,
             alpha, rng, planning):
    if strategy == 'degree':
        return {v: model.graph.degree(v) for v in candidates}
    if strategy == 'ar':
        return {v: ar[v] for v in candidates}
    if strategy == 'local' and state is not None:
        return local_benefit(model, candidates, state)
    if strategy != 'hybrid':
        raise ValueError('Unknown strategy')
    if alpha == 1:
        return calculate_hybrid({v: ar[v] for v in candidates},
                                dict.fromkeys(candidates, 0.), alpha)
    cib = estimate_cib(model, candidates, blocked, state,
                       seed=int(rng.integers(2**63)), activities=activities, **planning)
    return calculate_hybrid({v: ar[v] for v in candidates}, cib, alpha)


def static_immunization(model, activities, budget, strategy='hybrid', alpha=.5,
                        seed=42, centrality='degree', ar=None, **planning):
    validate_budget(budget)
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    ar = calculate_ar(model.graph, activities, centrality) if ar is None else ar
    blocked, rng = set(), np.random.default_rng(seed)
    for _ in range(min(budget, len(model.nodes))):
        candidates = set(model.nodes) - blocked
        scores = decision(model, activities, candidates, blocked, None, strategy,
                          ar, alpha, rng, planning)
        blocked.add(choose(scores))
    return blocked


def dynamic_immunization(model, activities, seeds, scenario, budget,
                         strategy='hybrid', alpha=.5, seed=42, per_step=1,
                         policy='all', centrality='degree', ar=None, **planning):
    validate_budget(budget, per_step)
    if policy not in ('all', 'frontier') or not 0 <= alpha <= 1:
        raise ValueError('Invalid policy or alpha')
    ar = calculate_ar(model.graph, activities, centrality) if ar is None else ar
    blocked, rng = set(), np.random.default_rng(seed)
    state = model.initial(seeds)
    while state.frontier:
        for _ in range(min(per_step, budget - len(blocked))):
            candidates = set(model.nodes) - blocked - state.infected
            if policy == 'frontier':
                neighbors = {v for u in state.frontier for v in model.graph.neighbors(u)}
                candidates &= neighbors
            if not candidates:
                break
            scores = decision(model, activities, candidates, blocked, state,
                              strategy, ar, alpha, rng, planning)
            blocked.add(choose(scores))
        state = model.step(state, blocked, scenario)
    return state, blocked
