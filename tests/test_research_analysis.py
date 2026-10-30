import unittest
import numpy as np
import pandas as pd

from experiment.research_analysis import paired_table, outbreak_pairs


class PairedAnalysisTests(unittest.TestCase):
    def test_outbreak_pairs_keep_identity_and_difference_direction(self):
        rows = []
        for trial in range(2):
            for method in ('MIG', 'CIB', 'Hybrid .5'):
                infected = 3 if method == 'MIG' else (2 if trial == 0 else 3)
                rows.append(dict(network='fixture', activity='exponential',
                                 beta=.5, budget_fraction=.1, trial=trial,
                                 mode='dynamic', method=method, plan_id=trial,
                                 evaluation_index=len(rows)+1,
                                 evaluation_seed=2**63 + trial,
                                 planning_seed=2**63 + 100 + len(rows),
                                 infected=infected, infection_fraction=infected/10,
                                 immunized=1, untreated_infected=5,
                                 waves=2, stratum='2+ waves'))
        pairs = outbreak_pairs(pd.DataFrame(rows))
        cib = pairs[pairs.treatment == 'CIB'].sort_values('trial')
        self.assertEqual(cib.outcome.tolist(), ['Win', 'Tie'])
        self.assertEqual(cib.delta_infected.tolist(), [-1, 0])
        self.assertEqual(cib.evaluation_seed_treatment.tolist(), [2**63, 2**63+1])
        changed = pd.DataFrame(rows)
        changed.loc[changed.method == 'CIB', 'evaluation_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'share the evaluation seed'):
            outbreak_pairs(changed)

    def test_static_intervals_use_five_paired_plan_means(self):
        rows = []
        for trial in range(100):
            plan = trial % 5
            for method in ('AR', 'CIB', 'Hybrid .5'):
                # A different plan-specific benefit, repeated over 20 outbreaks.
                value = .5 if method == 'AR' else .5 - plan*.01
                rows.append(dict(network='fixture', activity='exponential',
                                 beta=.5, budget_fraction=.1, trial=trial,
                                 mode='static', method=method, plan_id=plan,
                                 evaluation_seed=2**63 + trial,
                                 infection_fraction=value))
        table = paired_table(pd.DataFrame(rows))
        cib = table[table.treatment == 'CIB'].iloc[0]
        self.assertEqual(cib.independent_units, 5)
        self.assertEqual(cib.paired_outbreaks, 100)
        self.assertAlmostEqual(cib.mean_delta_pp, -2)
        half = 2.776*np.std([0, -1, -2, -3, -4], ddof=1)/np.sqrt(5)
        self.assertAlmostEqual(cib.ci95_high_pp, -2 + half)
        self.assertEqual(cib.wins, 80)
        self.assertEqual(cib.ties, 20)


if __name__ == '__main__':
    unittest.main()
