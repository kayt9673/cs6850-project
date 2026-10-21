
"""Fair evaluation with identical potential outbreak seeds for all methods."""

import numpy as np
import pandas as pd
from diffusion import simulate_cascade


def evaluate_immunization(
    graph, activities, beta, immunized, trial_seeds, random_seeds
):
    """Seeds are sampled from original graph, before immunization.

    If a sampled seed is immunized, that outbreak causes zero infections.
    """
    if len(trial_seeds) != len(random_seeds):
        raise ValueError(
            'trial_seeds and random_seeds must have equal length'
        )

    n = graph.number_of_nodes()
    fractions = []

    for source, seed in zip(trial_seeds, random_seeds):
        rng = np.random.default_rng(int(seed))

        infected = simulate_cascade(
            graph,
            activities,
            beta,
            [source],
            immunized=immunized,
            rng=rng
        )

        fractions.append(len(infected) / n)

    return float(np.mean(fractions))


def make_evaluation_trials(graph, trials=1000, seed=123):
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())

    sources = rng.choice(
        nodes, size=trials, replace=True
    ).tolist()

    random_seeds = rng.integers(
        0, 2**32 - 1, size=trials
    ).tolist()

    return sources, random_seeds


def compare_strategies(
    graph, activities, beta, strategies, trials=1000, seed=123
):
    sources, random_seeds = make_evaluation_trials(
        graph, trials, seed
    )

    rows = []

    for name, immunized in strategies.items():
        fraction = evaluate_immunization(
            graph,
            activities,
            beta,
            immunized,
            sources,
            random_seeds
        )

        rows.append({
            'strategy': name,
            'immunized_nodes': len(immunized),
            'mean_infected_fraction': fraction
        })

    return pd.DataFrame(rows).sort_values(
        'mean_infected_fraction'
    ).reset_index(drop=True)
