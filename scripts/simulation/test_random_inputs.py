import copy
import unittest
from random_inputs import resolve_random_inputs, final_seed

class ExplicitInputsTests(unittest.TestCase):
    def test_final_seeds_are_external_distinct_and_ignore_uuid_ports_and_order(self):
        config = {'models':['vgg19','resnet50'], 'nodes':[
            {'name':'full','template_name':'full','gpus':1,'port':9999,'share_targets':['1g.10gb']}]}
        first = resolve_random_inputs(config,101,2,2)
        other = copy.deepcopy(config); other['models'].reverse(); other['nodes'][0]['port']=1234
        self.assertEqual(first, resolve_random_inputs(other,101,2,2))
        self.assertNotEqual(first, resolve_random_inputs(config,102,2,2))
        schedules = first['worker_noise']
        self.assertNotEqual(schedules['full/0/1g.10gb/0']['resnet50'], schedules['full/0/1g.10gb/1']['resnet50'])
        self.assertEqual(schedules['full/0/1g.10gb/0']['resnet50'][0], final_seed(101,'noise/full/0/1g.10gb/0/resnet50/0'))
        rounding = 'rounding/full/0/1g.10gb/0/resnet50'
        self.assertEqual(first['seeds'][rounding], final_seed(101, rounding))
        self.assertNotEqual(first['seeds'][rounding], first['seeds']['rounding/full/0/1g.10gb/1/resnet50'])
        for bad in [-1,2**64,True]:
            with self.assertRaises(ValueError): resolve_random_inputs(config,bad)

if __name__=='__main__': unittest.main()
