"""Activity-dependent IC with explicit history and paired live-edge scenarios."""
from dataclasses import dataclass
import math
import numpy as np


def node_key(v):
    return (type(v).__name__, repr(v))


@dataclass
class InfectionState:
    infected: set
    frontier: set
    steps: int = 0

    def copy(self):
        return InfectionState(set(self.infected), set(self.frontier), self.steps)


class CascadeModel:
    def __init__(self, graph, activities, beta):
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError('Expected an undirected simple graph')
        if not 0 <= beta <= 1:
            raise ValueError('beta must be in [0, 1]')
        if any(v not in activities or not math.isfinite(activities[v]) or
               not 0 <= activities[v] <= 1 for v in graph):
            raise ValueError('Every node needs a finite activity in [0, 1]')
        self.graph = graph
        self.nodes = sorted(graph, key=node_key)
        self.arcs = [(u, v) for u in self.nodes
                     for v in sorted(graph.neighbors(u), key=node_key) if u != v]
        self.probabilities = np.array([beta * activities[u] * activities[v]
                                       for u, v in self.arcs])

    def scenario(self, rng):
        return {arc for arc, live in zip(self.arcs,
                rng.random(len(self.arcs)) < self.probabilities) if live}

    def initial(self, seeds, immunized=()):
        seeds, blocked = set(seeds), set(immunized)
        if not (seeds | blocked) <= set(self.nodes):
            raise ValueError('Unknown seed or immunized node')
        return InfectionState(seeds - blocked, seeds - blocked)

    def step(self, state, immunized, scenario):
        blocked = set(immunized)
        if blocked & state.infected or not state.frontier <= state.infected:
            raise ValueError('Cannot immunize infected nodes or use invalid history')
        new = {v for u in state.frontier for v in self.graph.neighbors(u)
               if v not in state.infected and v not in blocked and (u, v) in scenario}
        return InfectionState(state.infected | new, new, state.steps + 1)

    def finish(self, state, immunized, scenario):
        state = state.copy()
        if set(immunized) & state.infected:
            raise ValueError('Cannot reverse an infection')
        while state.frontier:
            state = self.step(state, immunized, scenario)
        return state


def simulate_cascade(graph, activities, beta, seeds, immunized=(), rng=None):
    model = CascadeModel(graph, activities, beta)
    rng = np.random.default_rng() if rng is None else rng
    return model.finish(model.initial(seeds, immunized), immunized,
                        model.scenario(rng)).infected
