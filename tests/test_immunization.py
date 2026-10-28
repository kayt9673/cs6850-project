import tempfile
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import networkx as nx
import numpy as np
from experiment.diffusion import CascadeModel, InfectionState, simulate_cascade
from experiment.immunization import (estimate_cib, static_immunization, dynamic_immunization,
                          modified_immunization_gain, sample_seeds)
from experiment.ranking import calculate_ar, calculate_hybrid, normalize_scores
from experiment.network import load_dataset
from experiment.experiments import stream_seed, run
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
        for strategy in ('ar', 'mig', 'hybrid'):
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

    def test_mig_measures_neighbor_risk_reduction(self):
        # Node 1 can be infected by 0 and then infect 2; node 2 has no
        # current exposure, so its downstream benefit is zero.
        scores = modified_immunization_gain(self.model, [1, 2], self.model.initial([0]))
        self.assertEqual(scores, {1: 1, 2: 0})
        state, protected = dynamic_immunization(self.model, self.activities,
            [0], self.live, 1, strategy='mig', mig_threshold=0)
        self.assertEqual(protected, {1})
        self.assertEqual(state.infected, {0})
        state, protected = dynamic_immunization(self.model, self.activities,
            [0], self.live, 1, strategy='mig', mig_threshold=1)
        self.assertEqual(protected, set())
        self.assertEqual(state.infected, set(self.graph))
        # A second route to neighbor 2 reduces the gain from blocking 1.
        triangle = nx.Graph([(0, 1), (0, 2), (1, 2)])
        model = CascadeModel(triangle, dict.fromkeys(triangle, 1.), .5)
        score = modified_immunization_gain(model, [1], model.initial([0]))
        self.assertAlmostEqual(score[1], .125)

    def test_cib_skips_nodes_outside_baseline_without_changing_score(self):
        # Candidate 3 is never reached by this one-edge live scenario.
        with patch.object(self.model, 'scenario', return_value={(0, 1)}):
            with patch('experiment.immunization.sample_seeds', return_value={0}):
                scores = estimate_cib(self.model, [1, 2, 3], samples=3)
        self.assertEqual(scores, {1: 1, 2: 0, 3: 0})

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

    def test_selected_plots_keep_all_aggregated_scenarios(self):
        row = dict(network='fixture', budget_fraction=.1, budget=1,
                   mode='static', strategy='hybrid', alpha=.5,
                   centrality='degree', mc_samples=2, seed_count=1,
                   seed_distribution='uniform', per_step=1,
                   candidate_policy='all', immunized=1, utilization=1.,
                   runtime_seconds=.01)
        raw = pd.DataFrame([dict(row, activity=activity, beta=beta,
                                 infection_fraction=.25)
                            for activity in ('exponential', 'power_law')
                            for beta in (.3, .5)])
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / 'raw.csv'
            raw.to_csv(raw_path, index=False)
            all_dir, selected_dir = Path(tmp) / 'all', Path(tmp) / 'selected'
            report(raw_path, all_dir)
            report(raw_path, selected_dir, plot_mode='selected')
            self.assertLess(len(list(selected_dir.glob('*.png'))),
                            len(list(all_dir.glob('*.png'))))
            self.assertEqual(len(pd.read_csv(all_dir / 'aggregated.csv')),
                             len(pd.read_csv(selected_dir / 'aggregated.csv')))

    def test_five_static_plans_are_reused_and_reported_as_five_plans(self):
        root = Path(__file__).resolve().parents[1]
        config = dict(datasets=[dict(name='karate',
            path=str(root / 'data/karate/edges.csv'),
            nodes_path=str(root / 'data/karate/nodes.csv'))],
            seed=42, activities=['exponential'], betas=[.5],
            budget_fractions=[.05], alphas=[0], mc_samples=2,
            trials=10, static_plans=5, checkpoint_rows=7)
        import os
        from experiment import experiments
        original = experiments.static_immunization
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                output_text = io.StringIO()
                with redirect_stdout(output_text):
                    with patch('experiment.experiments.static_immunization', wraps=original) as planner:
                        with patch('experiment.reporting.report'):
                            run(config)
                self.assertIn('Progress: 0.00%', output_text.getvalue())
                self.assertIn('Progress: 100.00% (60/60 evaluations)', output_text.getvalue())
                self.assertIn('Run complete.', output_text.getvalue())
                # Three static methods, each planned five times.
                self.assertEqual(planner.call_count, 15)
                raw = pd.read_csv(Path('output/1/raw.csv'))
                static = raw[raw['mode'] == 'static']
                self.assertEqual(len(static), 30)
                self.assertTrue((static.groupby(['strategy', 'alpha'], dropna=False)['plan_id']
                                 .nunique() == 5).all())
                summary = aggregate(raw)
                self.assertTrue((summary[summary['mode'] == 'static']['plans'] == 5).all())
                self.assertTrue((summary[summary['mode'] == 'dynamic']['plans'] == 10).all())
            finally:
                os.chdir(previous)

    def test_static_interval_uses_plan_means(self):
        row = dict(network='fixture', activity='exponential', beta=.5,
                   budget_fraction=.1, budget=1, mode='static', strategy='hybrid',
                   alpha=.5, centrality='degree', mc_samples=2, seed_count=1,
                   seed_distribution='uniform', per_step=1, candidate_policy='all',
                   immunized=1, utilization=1., runtime_seconds=.01)
        raw = pd.DataFrame([dict(row, plan_id=plan, infection_fraction=float(plan == 4))
                            for plan in range(5) for _ in range(20)])
        summary = aggregate(raw).iloc[0]
        self.assertEqual(summary['trials'], 100)
        self.assertEqual(summary['plans'], 5)
        self.assertAlmostEqual(summary['mean'], .2)
        self.assertAlmostEqual(summary['ci95_halfwidth'],
                               2.776 * np.sqrt(.2) / np.sqrt(5))

    def test_interrupted_run_resumes_without_duplicate_results(self):
        root = Path(__file__).resolve().parents[1]
        config = dict(datasets=[dict(name='karate',
            path=str(root / 'data/karate/edges.csv'),
            nodes_path=str(root / 'data/karate/nodes.csv'))],
            seed=42, activities=['exponential'], betas=[.5],
            budget_fractions=[.05], alphas=[0], mc_samples=2,
            trials=4, static_plans=2, checkpoint_rows=3,
            plot_mode='selected')
        import os
        from experiment import experiments
        original = experiments.sample_seeds
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 11:
                raise KeyboardInterrupt
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                with redirect_stdout(io.StringIO()), patch('experiment.reporting.report'):
                    with patch('experiment.experiments.sample_seeds', side_effect=interrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            run(config)
                path = Path('output/1/raw.csv')
                self.assertEqual(len(pd.read_csv(path)), 9)
                with path.open('ab') as stream:
                    stream.write(b'incomplete,last,row')
                with redirect_stdout(io.StringIO()), patch('experiment.reporting.report'):
                    run(config, resume='output/1')
                resumed = pd.read_csv(path)
                self.assertEqual(len(resumed), 24)
                with redirect_stdout(io.StringIO()), patch('experiment.reporting.report'):
                    run(config)
                fresh = pd.read_csv('output/2/raw.csv')
                unstable = ['planning_seconds', 'runtime_seconds']
                pd.testing.assert_frame_equal(resumed.drop(columns=unstable),
                                              fresh.drop(columns=unstable))
                with redirect_stdout(io.StringIO()):
                    with patch('experiment.reporting.report', side_effect=RuntimeError('plot interrupted')):
                        with self.assertRaisesRegex(RuntimeError, 'plot interrupted'):
                            run(config)
                report_path = Path('output/3/raw.csv')
                self.assertEqual(len(pd.read_csv(report_path)), 24)
                with redirect_stdout(io.StringIO()):
                    with patch('experiment.reporting.report') as report_mock:
                        run(config, resume='output/3')
                report_mock.assert_called_once()
                self.assertEqual(len(pd.read_csv(report_path)), 24)
                changed = dict(config, betas=[.3])
                with self.assertRaisesRegex(ValueError, 'Resume manifest differs'):
                    run(changed, resume='output/1')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
