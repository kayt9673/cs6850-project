
"""Activity-dependent independent-cascade diffusion."""
import numpy as np


def simulate_cascade(graph, activities, beta, seeds, immunized=(), rng=None):
    """Each newly infected node attempts each susceptible neighbor once."""
    if not 0 <= beta <= 1:
        raise ValueError('beta must be in [0, 1]')

    rng = np.random.default_rng() if rng is None else rng

    blocked = set(immunized)
    infected = set(seeds) - blocked
    frontier = set(infected)

    while frontier:
        next_frontier = set()

        for u in sorted(frontier):
            for v in sorted(graph.neighbors(u)):
                if v in infected or v in blocked or v in next_frontier:
                    continue

                p = beta * activities[u] * activities[v]

                if rng.random() < p:
                    next_frontier.add(v)

        infected.update(next_frontier)
        frontier = next_frontier

    return infected
