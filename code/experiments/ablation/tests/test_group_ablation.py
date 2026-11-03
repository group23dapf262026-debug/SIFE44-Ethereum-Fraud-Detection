"""Check column deletion semantics and that all models obey one declared matrix."""
import tempfile
import unittest
import numpy as np
import torch
from sife20.features import FEATURE_GROUPS,ARMS,ARM_REMOVED_GROUP,selected_sife_indices,select_inputs,PURE36_INDICES
from sife20.models import ARM_WIDTHS,arm_columns,build_classifier
from sife20.tabular_inputs import tabular_input_frame
from sife20.protocol import open_study,MODELS,supported_arms
from test_study import fixture

class GroupAblationChecks(unittest.TestCase):
    def test_groups_partition44_and_widths_are_declared(self):
        groups=list(FEATURE_GROUPS.values())
        self.assertEqual([len(g) for g in groups],[20,8,8,8])
        flattened=[i for g in groups for i in g]
        self.assertEqual(sorted(flattened),list(range(44)))
        self.assertEqual(len(flattened),len(set(flattened)))
        self.assertEqual([ARM_WIDTHS[a] for a in ARMS],[58,38,50,50,50])
        self.assertEqual(selected_sife_indices('X14_Pure36'),PURE36_INDICES)

    def test_independent_explicit_column_deletions_all_models(self):
        source=np.arange(42,dtype=float).reshape(3,14)-10
        values=np.arange(132,dtype=float).reshape(3,44)
        removed={
            'X14_SIFE44':[],
            'without_distribution20':[0,1,2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25],
            'without_flow8':list(range(28,36)),
            'without_counterparty8':list(range(36,44)),
            'X14_Pure36':[5,6,12,13,19,20,26,27]}
        for arm in ARMS:
            expected=np.column_stack((source,np.delete(values,removed[arm],axis=1)))
            for model in MODELS:
                np.testing.assert_array_equal(select_inputs(source,values,arm,model),expected)
                np.testing.assert_array_equal(tabular_input_frame(source,values,arm,model).to_numpy(),expected)
            full=np.column_stack((source,values))
            np.testing.assert_array_equal(full[:,list(arm_columns(arm))],expected)

    def test_removed_columns_cannot_change_selected_inputs(self):
        source=np.zeros((2,14));values=np.arange(88,dtype=float).reshape(2,44)
        for arm in ARMS[1:]:
            changed=values.copy();changed[:,FEATURE_GROUPS[ARM_REMOVED_GROUP[arm]]]=999999.
            np.testing.assert_array_equal(select_inputs(source,changed,arm),select_inputs(source,values,arm))

    def test_all_five_have_same_five_arms_and5000fit_budget(self):
        total=0
        for dataset in ('Phish','Ponzi'):
            for model in MODELS:
                with tempfile.TemporaryDirectory() as folder:
                    study=open_study(fixture(dataset),folder,dataset,model,'cpu',{'test':'budget'})
                    p=study['protocol'];self.assertEqual(list(p['arms']),list(ARMS))
                    self.assertEqual(p['control_arm'],'X14_SIFE44')
                    self.assertEqual(p['study'],'sife20_group_ablation_five_fixed_v1')
                    self.assertFalse(p['hyperparameter_search'])
                    self.assertEqual(p['model_seeds'],[12,13,14,15,16])
                    total+=len(p['split_seeds'])*len(p['model_seeds'])*len(p['arms'])
        self.assertEqual(total,5000)

    def test_all_neural_arm_widths_are_actual_model_inputs(self):
        data=fixture();stats=dict(mean=[0.]*58,scale=[1.]*58)
        for model in ('gat','sage','metaifd'):
            for arm in ARMS:
                net=build_classifier(model,arm,{'hidden':8,'views':3},data['graph'],'EOA',stats)
                if model=='gat':self.assertEqual(net.conv1.in_channels,ARM_WIDTHS[arm])
                elif model=='sage':self.assertEqual(net.input_encoder.in_features,ARM_WIDTHS[arm])
                else:self.assertEqual(net.lin_dict['EOA'].in_channels,ARM_WIDTHS[arm])

if __name__=='__main__':unittest.main()
