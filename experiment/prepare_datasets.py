"""Download and prepare the four selected networks without running experiments."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile
import networkx as nx

SOURCES = {
    'karate': ('https://nrvis.com/download/data/misc/karate.zip',
               'https://networkrepository.com/karate.php', 34, 78),
    'dolphins': ('https://nrvis.com/download/data/misc/dolphins.zip',
                 'https://networkrepository.com/dolphins.php', 62, 159),
    'football': ('https://networks.skewed.de/net/football/files/football.csv.zip',
                 'https://networks.skewed.de/net/football', 115, 613),
    'netscience': ('https://networks.skewed.de/net/netscience/files/netscience.csv.zip',
                   'https://networks.skewed.de/net/netscience', 1589, 2742),
}


def matrix_market(text):
    """Read the symmetric pattern Matrix Market files supplied by NR."""
    lines = iter(text.splitlines())
    if next(lines).strip().lower() != '%%matrixmarket matrix coordinate pattern symmetric':
        raise ValueError('Expected a symmetric pattern Matrix Market graph')
    entries = [line.split() for line in lines if line.strip() and not line.startswith('%')]
    n, columns, count = map(int, entries[0])
    if n != columns or len(entries) - 1 != count:
        raise ValueError('Invalid Matrix Market dimensions or entry count')
    graph = nx.Graph()
    graph.add_nodes_from(str(i) for i in range(1, n + 1))
    for row in entries[1:]:
        u, v = map(int, row)
        if not (1 <= u <= n and 1 <= v <= n):
            raise ValueError('Matrix Market node index out of bounds')
        graph.add_edge(str(u), str(v))
    return graph


def csv_rows(text):
    # Netzschleuder prefixes headers with '# ' and separates records by blank lines.
    reader = csv.DictReader(io.StringIO(text.lstrip('# ')), skipinitialspace=True)
    for row in reader:
        yield {key.strip(): value for key, value in row.items()}


def catalog_csv(archive):
    graph = nx.Graph()
    for row in csv_rows(archive.read('nodes.csv').decode('utf-8-sig')):
        node = row.pop('index')
        graph.add_node(node, **row)
    for row in csv_rows(archive.read('edges.csv').decode('utf-8-sig')):
        source, target = row.pop('source'), row.pop('target')
        if source not in graph or target not in graph:
            raise ValueError('Unknown node in edge CSV')
        graph.add_edge(source, target, **row)
    return graph


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prepare(root='data', download=False):
    root = Path(root)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, page, nodes, edges) in SOURCES.items():
        archive_path = raw / url.rsplit('/', 1)[-1]
        if not archive_path.exists():
            if not download:
                raise FileNotFoundError(f'{archive_path}: rerun with --download')
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt download')
            archive_path.write_bytes(payload)
        with zipfile.ZipFile(archive_path) as archive:
            graph = (matrix_market(archive.read(f'{name}.mtx').decode())
                     if name in ('karate', 'dolphins') else catalog_csv(archive))
        if (len(graph), graph.number_of_edges()) != (nodes, edges):
            raise ValueError(f'Unexpected {name} size: {len(graph)}, {graph.number_of_edges()}')
        if nx.number_of_selfloops(graph):
            raise ValueError(f'Unexpected self-loop in {name}')
        destination = root / name
        destination.mkdir(exist_ok=True)
        node_keys = sorted({k for _, attrs in graph.nodes(data=True) for k in attrs})
        edge_keys = sorted({k for _, _, attrs in graph.edges(data=True) for k in attrs})
        write_csv(destination / 'nodes.csv',
                  [dict(node=v, **attrs) for v, attrs in graph.nodes(data=True)], ['node', *node_keys])
        write_csv(destination / 'edges.csv',
                  [dict(source=u, target=v, **attrs) for u, v, attrs in graph.edges(data=True)],
                  ['source', 'target', *edge_keys])
        manifest[name] = dict(source=page, download=url, nodes=nodes, edges=edges,
            archive=str(archive_path.relative_to(root)),
            sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            transformations='Undirected simple graph; all nodes/components retained; attributes preserved; '
                            'edge weights unused by activity-based propagation; no empirical activities supplied.')
        print(f'{name}: {nodes} nodes, {edges} edges')
    (root / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download missing source archives')
    parser.add_argument('--output', default='data')
    args = parser.parse_args()
    prepare(args.output, args.download)
