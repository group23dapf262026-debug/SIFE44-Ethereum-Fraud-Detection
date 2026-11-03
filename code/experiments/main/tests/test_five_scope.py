"""Prevent accidental dispatch of removed models or altered fixed study budgets."""
import tempfile
import unittest
from sife20.protocol import MODELS,open_study,fixed_configuration
from test_study import fixture


class FiveModelScopeChecks(unittest.TestCase):
    def test_removed_lr_cannot_be_opened_as_main_run(self):
        self.assertEqual(set(MODELS),{'metaifd','gat','sage','lightgbm','rf'})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):open_study(fixture(),folder,'Phish','lr','cpu',{})

    def test_fixed_configs_match_current_reference(self):
        expected={'metaifd':dict(hidden=128,lr=.001,views=3,contrast_weight=.1),
                  'gat':dict(hidden=32,lr=.01),'sage':dict(hidden=32,lr=.01),
                  'lightgbm':dict(n_estimators=300,lr=.01,num_leaves=31),
                  'rf':dict(n_estimators=100,max_depth=10)}
        for model in MODELS:self.assertEqual(fixed_configuration(model),expected[model])


if __name__=='__main__':unittest.main()
