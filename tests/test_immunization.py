import tempfile
import unittest
from pathlib import Path
import networkx as nx
import numpy as np
from experiment.diffusion import CascadeModel, InfectionState, simulate_cascade
from experiment.immunization import (estimate_cib, static_immunization, dynamic_immunization,
                          local_benefit, sample_seeds)
from experiment.ranking import calculate_ar, calculate_hybrid, normalize_scores
from experiment.network import load_dataset
from experiment.experiments import stream_seed
from unittest.mock import patch
import pandas as pd
from experiment.reporting import aggregate, report


class ImmunizationTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.path_graph(4)
        self.activities = dict.fromkeys(self.graph, 1.)
        self.model = CascadeModel(self.graph, self.activities, 1.)
        self.live = set(self.model.arcs)

    def test_zero_and_unit_transmission(self):
        self.assertEqual(simulate_cascade(self.graph, self.activities, 0, [0]), {0})
        self.assertEqual(simulate_cascade(self.graph, self.activities, 1, [0]), set(range(4)))

    def test_immunized_seed_and_bridge(self):
        self.assertEqual(simulate_cascade(self.graph, self.activities, 1, [0], [0]), set())
        self.assertEqual(simulate_cascade(self.graph, self.activities, 1, [0], [1]), {0})

    def test_frontier_history_no_second_attempt(self):
        state = self.model.step(self.model.initial([0]), set(), set())
        self.assertFalse(state.frontier)
        self.assertEqual(self.model.finish(state, set(), self.live).infected, {0})
        state = InfectionState({0, 1}, {1}, 1)
        # Old node 0 must not transmit again even with a newly live arc.
        graph = nx.Graph([(0, 1), (0, 2)])
        model = CascadeModel(graph, dict.fromkeys(graph, 1), 1)
        self.assertEqual(model.finish(state, set(), set(model.arcs)).infected, {0, 1})

    def test_cannot_reverse_infection(self):
        with self.assertRaises(ValueError):
            self.model.finish(self.model.initial([0]), {0}, self.live)
        with self.assertRaises(ValueError):
            estimate_cib(self.model, [0], state=self.model.initial([0]))

    def test_hand_calculated_dynamic_cib(self):
        scores = estimate_cib(self.model, [1, 2, 3], state=self.model.initial([0]), samples=5)
        for v, expected in [(1, 3), (2, 2), (3, 1)]:
            self.assertAlmostEqual(scores[v], expected)
        ended = InfectionState({0}, set())
        self.assertEqual(estimate_cib(self.model, [3], state=ended, samples=2), {3: 0})

    def test_nonnegative_and_reproducible(self):
        model = CascadeModel(self.graph, self.activities, .5)
        a = estimate_cib(model, list(self.graph), samples=50, seed=7)
        self.assertEqual(a, estimate_cib(model, list(self.graph), samples=50, seed=7))
        self.assertTrue(all(v >= -1e-12 for v in a.values()))
        # Seed 0/1/2/3 prevents 3/4/2/2 infections: expectation 11/4.
        scores = estimate_cib(self.model, [1], samples=5000, seed=9)
        self.assertAlmostEqual(scores[1], 2.75, delta=.05)

    def test_budget_and_order(self):
        protected = static_immunization(self.model, self.activities, 2, samples=4)
        self.assertEqual(len(protected), 2)
        for strategy in ('ar', 'local', 'hybrid'):
            state, protected = dynamic_immunization(self.model, self.activities, [0], self.live,
                                                    2, strategy=strategy, samples=4, policy='frontier')
            self.assertLessEqual(len(protected), 2)
            self.assertIn(0, state.infected)
            self.assertFalse(protected & state.infected)
            self.assertEqual(state.infected, {0})  # immunization before transmission

    def test_early_termination(self):
        state, protected = dynamic_immunization(self.model, self.activities, [0], set(),
                                                4, strategy='ar')
        self.assertEqual(len(protected), 1)
        self.assertEqual(state.steps, 1)
        state, protected = dynamic_immunization(self.model, self.activities, [], self.live, 4)
        self.assertFalse(protected)

    def test_static_has_no_evaluation_state(self):
        import inspect
        signature = inspect.signature(static_immunization)
        self.assertNotIn('scenario', signature.parameters)
        self.assertNotIn('state', signature.parameters)
        a = static_immunization(self.model, self.activities, 2, samples=5, seed=3)
        # Evaluating arbitrary futures cannot alter the static decision.
        self.model.finish(self.model.initial([3], a), a, self.live)
        self.assertEqual(a, static_immunization(self.model, self.activities, 2, samples=5, seed=3))

    def test_ar_formula_and_zero(self):
        ar = calculate_ar(self.graph, self.activities)
        self.assertAlmostEqual(ar[1], (2/3)**2 * np.log(4))
        self.assertEqual(calculate_ar(self.graph, dict.fromkeys(self.graph, 0)), dict.fromkeys(self.graph, 0))
        activity = dict(self.activities)
        activity[0] = 0
        self.assertTrue(np.isfinite(calculate_ar(self.graph, activity)[0]))
        for centrality in ('degree', 'betweenness', 'k_core', 'eigenvector'):
            self.assertEqual(set(calculate_ar(self.graph, self.activities, centrality)), set(self.graph))

    def test_hybrid_ablations(self):
        ar, cib = {0: 4., 1: 2.}, {0: 1., 1: 3.}
        self.assertEqual(calculate_hybrid(ar, cib, 0), normalize_scores(cib))
        self.assertEqual(calculate_hybrid(ar, cib, 1), normalize_scores(ar))
        self.assertEqual(normalize_scores({0: 0}), {0: 0})

    def test_ar_manual_unequal_activity(self):
        # Three-node path: degree centralities 1/2, 1, 1/2; total activity 1.
        graph = nx.path_graph(3)
        activities = {0: .125, 1: .375, 2: .5}
        expected = {0: .25 * np.log(8), 1: np.log(8/3), 2: .25 * np.log(2)}
        actual = calculate_ar(graph, activities)
        for node in graph:
            self.assertAlmostEqual(actual[node], expected[node])

    def test_ar_log_base_cancels_in_hybrid(self):
        ar = calculate_ar(self.graph, self.activities)
        base2 = {v: score / np.log(2) for v, score in ar.items()}
        cib = {0: 3., 1: 1., 2: 4., 3: 2.}
        for alpha in (0, .25, .5, .75, 1):
            natural = calculate_hybrid(ar, cib, alpha)
            binary = calculate_hybrid(base2, cib, alpha)
            for v in ar:
                self.assertAlmostEqual(natural[v], binary[v])

    def test_ar_dependent_strategies_use_squared_centrality(self):
        # Unsquared centrality would prefer endpoint 0; squared prefers node 1.
        graph = nx.path_graph(3)
        activities = {0: .1, 1: .3, 2: .6}
        model = CascadeModel(graph, activities, 1)
        for strategy, alpha in (('ar', .5), ('hybrid', 1)):
            protected = static_immunization(model, activities, 1,
                                            strategy=strategy, alpha=alpha)
            self.assertEqual(protected, {1})
            _, protected = dynamic_immunization(model, activities, [2], set(), 1,
                                                strategy=strategy, alpha=alpha)
            self.assertEqual(protected, {1})

    def test_local_immediate_only(self):
        scores = local_benefit(self.model, [1, 2], self.model.initial([0]))
        self.assertEqual(scores, {1: 1, 2: 0})

    def test_validation_and_seed_streams(self):
        with self.assertRaises(ValueError):
            CascadeModel(self.graph, self.activities, 2)
        with self.assertRaises(ValueError):
            static_immunization(self.model, self.activities, -1)
        self.assertNotEqual(stream_seed(42, 'planning'), stream_seed(42, 'evaluation'))
        seeds = sample_seeds(self.model, np.random.default_rng(1), 2)
        self.assertEqual(len(seeds), 2)

    def test_loader_preserves_ids_isolates_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            edges, nodes = Path(tmp)/'edges.csv', Path(tmp)/'nodes.csv'
            edges.write_text('source,target,weight\n001,002,0.4\n')
            nodes.write_text('node,activity\n001,0.2\n002,0.5\n003,0.7\n')
            graph = load_dataset(dict(path=str(edges), nodes_path=str(nodes)))
            self.assertEqual(set(graph), {'001', '002', '003'})
            self.assertEqual(graph['001']['002']['weight'], '0.4')
            self.assertEqual(float(graph.nodes['003']['activity']), .7)

    def test_dynamic_repeat_and_multiple_per_step(self):
        args = (self.model, self.activities, [0], self.live, 3)
        first = dynamic_immunization(*args, per_step=2, samples=4, seed=9)
        self.assertEqual(first, dynamic_immunization(*args, per_step=2, samples=4, seed=9))
        self.assertEqual(len(first[1]), 2)

    def test_planning_does_not_receive_evaluation_future(self):
        from experiment import immunization
        original = immunization.estimate_cib
        observed = []
        def spy(*args, **kwargs):
            observed.append((args, kwargs))
            return original(*args, **kwargs)
        with patch('experiment.immunization.estimate_cib', side_effect=spy):
            dynamic_immunization(self.model, self.activities, [0], self.live,
                                  1, samples=2)
        self.assertTrue(observed)
        self.assertTrue(all('scenario' not in kw for _, kw in observed))
        self.assertIsInstance(observed[0][0][3], InfectionState)

    def test_report_from_fixture_not_an_experiment(self):
        row = dict(network='fixture', activity='empirical', beta=.5,
                   budget_fraction=.25, budget=1, mode='static', strategy='hybrid',
                   alpha=.5, centrality='degree', mc_samples=2, seed_count=1,
                   seed_distribution='uniform', per_step=1, candidate_policy='all',
                   immunized=1, utilization=1., runtime_seconds=.01)
        raw = pd.DataFrame([dict(row, infection_fraction=x) for x in (.25, .75)])
        summary = aggregate(raw)
        self.assertEqual(summary.loc[0, 'mean'], .5)
        self.assertAlmostEqual(summary.loc[0, 'std'], np.sqrt(.125))
        self.assertEqual(summary.loc[0, 'trials'], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.csv'
            raw.to_csv(path, index=False)
            report(path, tmp)
            self.assertTrue((Path(tmp) / 'aggregated.csv').exists())
            self.assertEqual(len(list(Path(tmp).glob('*.png'))), 5)


if __name__ == '__main__':
    unittest.main()
