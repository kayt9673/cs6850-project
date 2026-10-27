"""Input-preparation tests; no immunization experiments are run."""
import io
import json
from pathlib import Path
import unittest
import zipfile
from experiment.network import load_dataset
from experiment.prepare_datasets import matrix_market, catalog_csv, SOURCES


class DatasetTests(unittest.TestCase):
    def test_matrix_market_keeps_isolates(self):
        graph = matrix_market('%%MatrixMarket matrix coordinate pattern symmetric\n% comment\n3 3 1\n2 1\n')
        self.assertEqual(set(graph), {'1', '2', '3'})
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(graph.degree('3'), 0)

    def test_catalog_csv_preserves_attributes_and_isolates(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('nodes.csv', '# index, label\n0,"A, B"\n\n1,C\n2,D\n')
            archive.writestr('edges.csv', '# source, target, value\n0,1,2.5\n')
        with zipfile.ZipFile(buffer) as archive:
            graph = catalog_csv(archive)
        self.assertEqual(graph.nodes['0']['label'], 'A, B')
        self.assertEqual(graph['0']['1']['value'], '2.5')
        self.assertEqual(graph.degree('2'), 0)

    def test_selected_datasets_load_in_both_configs(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('validation', 'full'):
            config = json.loads((root / 'configs' / (name + '.json')).read_text())
            self.assertEqual({spec['name'] for spec in config['datasets']}, set(SOURCES))
            self.assertNotIn('empirical', config['activities'])
            for spec in config['datasets']:
                local = dict(spec, path=str(root / spec['path']), nodes_path=str(root / spec['nodes_path']))
                graph = load_dataset(local)
                expected = SOURCES[spec['name']][2:]
                self.assertEqual((len(graph), graph.number_of_edges()), expected)
                self.assertFalse(graph.is_directed())
