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
    candidate_set = set(candidates)
    if not candidate_set <= set(model.nodes) - blocked - (state.infected if state else set()):
        raise ValueError('CIB candidates must be susceptible and non-immunized')
    rng = np.random.default_rng(seed)
    scores = dict.fromkeys(candidates, 0.0)
    for _ in range(samples):
        live = model.scenario(rng)
        sources = (sample_seeds(model, rng, seed_count, seed_distribution, activities)
                   if state is None else None)
        initial = model.initial(sources, blocked) if state is None else state
        baseline_state = model.finish(initial, blocked, live)
        baseline = len(baseline_state.infected)
        # Blocking a node outside this live-edge cascade cannot change it.
        for v in candidate_set & baseline_state.infected:
            protected = blocked | {v}
            initial = model.initial(sources, protected) if state is None else state
            scores[v] += (baseline - len(model.finish(initial, protected, live).infected)) / samples
    return scores


def modified_immunization_gain(model, candidates, state, immunized=()):
    """Shang et al. MIG (Eqs. 6-8), adapted to this model's p_uv.

    AP(v) is the chance v is active for the next transmission opportunity:
    one for the current infectious frontier, zero for earlier infected nodes,
    and 1-prod(1-p_uv) for susceptible nodes exposed by that frontier.
    MIG(i) sums the reduction in each susceptible neighbor j's infection risk
    when i is immunized. The paper's community-based p_uv is replaced by the
    common activity-dependent p_uv used by every strategy here.
    """
    blocked = set(immunized)
    susceptible = set(model.nodes) - state.infected - blocked
    if not set(candidates) <= susceptible:
        raise ValueError('MIG candidates must be susceptible and non-immunized')
    probabilities = dict(zip(model.arcs, model.probabilities))
    activation = {}
    for v in model.nodes:
        if v in state.frontier:
            activation[v] = 1.0
        elif v in state.infected or v in blocked:
            activation[v] = 0.0
        else:
            activation[v] = 1 - np.prod([1 - probabilities[(u, v)]
                                          for u in model.graph.neighbors(v)
                                          if u in state.frontier])
    scores = {}
    for i in candidates:
        gain = 0.0
        for j in model.graph.neighbors(i):
            if j not in susceptible:
                continue
            other_failure = np.prod([1 - activation[v] * probabilities[(v, j)]
                                     for v in model.graph.neighbors(j) if v != i])
            gain += activation[i] * probabilities[(i, j)] * other_failure
        scores[i] = float(gain)
    return scores


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
    if strategy == 'mig' and state is not None:
        return modified_immunization_gain(model, candidates, state, blocked)
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
                         policy='all', centrality='degree', ar=None,
                         mig_threshold=0.0, **planning):
    validate_budget(budget, per_step)
    if policy not in ('all', 'frontier') or not 0 <= alpha <= 1:
        raise ValueError('Invalid policy or alpha')
    if not np.isfinite(mig_threshold) or mig_threshold < 0:
        raise ValueError('MIG threshold must be finite and nonnegative')
    ar = calculate_ar(model.graph, activities, centrality) if ar is None else ar
    blocked, rng = set(), np.random.default_rng(seed)
    state = model.initial(seeds)
    while state.frontier:
        for _ in range(min(per_step, budget - len(blocked))):
            candidates = set(model.nodes) - blocked - state.infected
            if policy == 'frontier' or strategy == 'mig':
                neighbors = {v for u in state.frontier for v in model.graph.neighbors(u)}
                candidates &= neighbors
            if not candidates:
                break
            scores = decision(model, activities, candidates, blocked, state,
                              strategy, ar, alpha, rng, planning)
            if strategy == 'mig' and max(scores.values()) <= mig_threshold:
                break
            blocked.add(choose(scores))
        state = model.step(state, blocked, scenario)
    return state, blocked
