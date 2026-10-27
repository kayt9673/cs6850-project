"""Load network datasets and generate synthetic node activities."""
import networkx as nx
import numpy as np


def generate_activities(graph, distribution='mixed', rng=None, floor=0.05):
    """Sample activities, normalize globally to [floor, 1].

    Distribution parameters are illustrative, not exact Li et al. replications.
    """
    rng = np.random.default_rng() if rng is None else rng
    nodes = list(graph.nodes())
    n = len(nodes)
    if not n:
        return {}

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


def load_dataset(spec):
    """Load string-ID CSV edges/nodes or GraphML, retaining source attributes.

    CSV edges require source,target; optional weight. Nodes CSV requires node
    and optionally activity; it also records isolates. Directed GraphML requires
    explicit undirected=true. Source files are never modified.
    """
    import csv
    from pathlib import Path
    path = Path(spec['path'])
    if path.suffix.lower() == '.graphml':
        graph = nx.read_graphml(path)
    else:
        graph = nx.Graph()
        with path.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                attrs = {k: v for k, v in row.items() if k not in ('source', 'target')}
                graph.add_edge(row['source'], row['target'], **attrs)
    if graph.is_directed() and not spec.get('undirected', False):
        raise ValueError('Directed input requires explicit undirected=true')
    # The propagation model uses topology, not input weights. The source is intact.
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if spec.get('nodes_path'):
        with open(spec['nodes_path'], newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                graph.add_node(row['node'], **{k: v for k, v in row.items()
                                              if k != 'node' and v != ''})
    if not graph:
        raise ValueError('Empty dataset')
    return graph
