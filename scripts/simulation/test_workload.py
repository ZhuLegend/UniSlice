import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run import scheduler_config
from workload import paper_workload, long_cycle_workload


class PaperWorkloadTest(unittest.TestCase):
    def test_long_cycle_has_five_switches_and_restores_load(self):
        models = ['vgg16', 'vgg19', 'resnet18', 'resnet50', 'alexnet']
        work, events = long_cycle_workload(models, 4, 2, 99)
        self.assertEqual((work, events), long_cycle_workload(models, 4, 2, 99))
        self.assertEqual(len(work['phases']), 6)
        self.assertEqual(work['duration_seconds'], 12)
        self.assertEqual(work['phases'][0]['rates'], work['initial_rates'])
        self.assertEqual(work['phases'][-1]['rates'], work['initial_rates'])
        self.assertEqual(work['phases'][3]['rates']['resnet18'], 100)
        self.assertEqual(work['phases'][3]['rates']['vgg16'], 50)
        self.assertEqual(work['phases'][4]['rates']['vgg16'], 100)
        self.assertTrue(all(0 <= t < 12 and m in models and 1 <= b <= 32 for t,m,b in events))
        self.assertEqual(events, sorted(events))

    def test_replay_is_seeded_and_startup_matches_first_segment(self):
        models = ['alexnet', 'vgg16', 'vgg19']
        work, events = paper_workload(models, 4, 600, 7)
        self.assertEqual((work, events), paper_workload(models, 4, 600, 7))
        self.assertEqual(work['initial_rates'], work['phases'][0]['rates'])
        self.assertEqual(work['phases'][1]['rates'], {'alexnet': 50, 'vgg16': 100, 'vgg19': 100})
        self.assertEqual(work['phases'][2]['rates'], work['initial_rates'])
        self.assertEqual(work['slo_ms'], [250, 400])
        self.assertEqual(work['rate_unit'], 'samples/s')
        self.assertEqual(work['batch_size_range'], [1,32])
        self.assertTrue(all(0 <= t < 1800 and m in models and 1 <= b <= 32 for t, m, b in events))
        self.assertLess(abs(sum(b for _,_,b in events)/len(events)-16.5), .5)
        self.assertEqual(events, sorted(events))
        # Catch sample/request confusion or an unintended cap on high rates.
        for phase, expected in [(0, 90000), (1, 150000), (2, 90000)]:
            samples = sum(b for t, _, b in events if phase*600 <= t < (phase+1)*600)
            self.assertLess(abs(samples-expected), expected*.06)

    def test_single_sample_control_and_invalid_ranges(self):
        work, events = paper_workload(['alexnet'], 8, 60, 7, (1,1))
        self.assertTrue(all(b == 1 for _,_,b in events))
        self.assertLess(abs(len(events)-18000), 18000*.05)
        for bounds in ((0,32), (2,1), (1,33)):
            with self.assertRaises(ValueError):
                paper_workload(['alexnet'], 1, batch_range=bounds)
        with patch('workload.MAX_REPLAY_EVENTS', 2), self.assertRaisesRegex(ValueError, 'memory ceiling'):
            paper_workload(['alexnet'], 32)
        with self.assertRaises(ValueError):
            paper_workload(['alexnet'], 1, phase_seconds=float('nan'))

    def test_paper_topology_and_scheduler_rates(self):
        config = json.loads((Path(__file__).parents[2]/'resource/simulation/paper-a100.json').read_text())
        nodes = config['nodes']
        self.assertEqual(sum(n['resource']['ce'] for n in nodes), 52)
        self.assertEqual(sum(n['resource']['ce'] == 7 for n in nodes), 4)
        self.assertEqual(sum(n['resource']['ce'] == 3 for n in nodes), 8)
        self.assertTrue(all(n['share_targets'] == ['1g.10gb'] for n in nodes))
        self.assertEqual([config[k] for k in ['create_delay_ms', 'destroy_delay_ms']], [15000, 15000])
        work, _ = paper_workload(config['models'], 8, 1)
        toml = scheduler_config(config, work['initial_rates'])
        for model in config['models']:
            self.assertIn(f'"{model}" = 100.0', toml)


if __name__ == '__main__':
    unittest.main()
