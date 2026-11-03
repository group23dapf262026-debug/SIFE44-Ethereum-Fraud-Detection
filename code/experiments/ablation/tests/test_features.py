"""Meaningful feature invariants and parity checks, not classifier benchmarks."""
import json
from pathlib import Path
import unittest

import numpy as np

from sife20.features import (CHANNELS,SIFE44_NAMES,PURE36_INDICES,INTENSITY8_INDICES,
                           extract_sife44,select_inputs,summarize_channel)

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    relations=[('CA','call','CA'),('EOA','call','CA'),('EOA','trans','CA'),
               ('CA','trans','CA'),('CA','trans','EOA'),('EOA','trans','EOA')]
    ix=np.array([[0,0,1,2,2,3],[1,2,2,0,3,1]],dtype=np.int64)
    attr=np.column_stack(([1.,2.,3.,4.,1.,2.],[0.,.5,2.,5.,1.,.3]))
    edges={et:dict(edge_index=ix.copy(),edge_attr=attr.copy()) for et in relations}
    return {'CA':4,'EOA':4},edges


class FeatureChecks(unittest.TestCase):
    def test_column_partition_and_arm_widths(self):
        self.assertEqual(len(SIFE44_NAMES),len(set(SIFE44_NAMES)))
        self.assertEqual(len(SIFE44_NAMES),44)
        self.assertEqual(len(PURE36_INDICES),36)
        self.assertEqual(set(PURE36_INDICES)|set(INTENSITY8_INDICES),set(range(44)))
        self.assertFalse(set(PURE36_INDICES)&set(INTENSITY8_INDICES))
        x=np.zeros((4,14));s=np.zeros((4,44))
        for arm,width in [('X14',14),('Pure36',36),('X14_Pure36',50),('X14_SIFE44',58)]:
            self.assertEqual(select_inputs(x,s,arm).shape,(4,width))
        self.assertEqual(select_inputs(None,s,'Pure36').shape,(4,36))
        with self.assertRaises(ValueError):select_inputs(x,s,'Pure36','metaifd')

    def test_absent_channel_is_zero(self):
        result=extract_sife44({'CA':2,'EOA':2},{},'EOA',np.array([0,1]))
        np.testing.assert_array_equal(result['sife44'],np.zeros((2,44)))

    def test_zero_mass_uses_documented_uniform_fallback(self):
        def records():yield np.array([0,0]),np.array([1.,3.]),np.zeros(2),True
        values,_=summarize_channel(1,records)
        np.testing.assert_allclose(values[0,[0,1,2,4]],[np.log(2),1,.5,.5])
        self.assertGreater(values[0,3],0)

    def test_single_partner_and_direction(self):
        edges={('CA','trans','EOA'):dict(edge_index=np.array([[0],[1]]),edge_attr=np.array([[2.,6.]]))}
        out=extract_sife44({'CA':2,'EOA':2},edges,'EOA',np.array([1]))['sife44'][0]
        self.assertEqual(out[16],1.)  # transfer-in HHI
        self.assertEqual(out[18],1.)  # transfer-in maximum share
        self.assertEqual(out[40],1.)  # transfer-in CA record share
        self.assertEqual(out[41],1.)  # transfer-in CA count share
        self.assertEqual(out[34],1.)  # transfer degree asymmetry
        self.assertEqual(out[35],1.)  # transfer value asymmetry
        np.testing.assert_array_equal(out[:14],np.zeros(14))

    def test_chunk_and_relation_order_do_not_change_semantics(self):
        counts,edges=fixture()
        a=extract_sife44(counts,edges,'CA',np.arange(4),1)['sife44']
        b=extract_sife44(counts,dict(reversed(list(edges.items()))),'CA',np.arange(4),100)['sife44']
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_pure36_positive_rescaling_and_shape_information(self):
        counts,edges=fixture()
        a=extract_sife44(counts,edges,'CA',np.arange(4))['sife44']
        for record in edges.values():record['edge_attr']*=np.array([7.,10.])
        b=extract_sife44(counts,edges,'CA',np.arange(4))['sife44']
        np.testing.assert_allclose(a[:,PURE36_INDICES],b[:,PURE36_INDICES],rtol=1e-12,atol=1e-12)
        self.assertFalse(np.allclose(a[:,INTENSITY8_INDICES],b[:,INTENSITY8_INDICES]))
        def summarize(values):
            def records():yield np.array([0,0]),np.ones(2),np.array(values),False
            return summarize_channel(1,records)[0][0,0]
        self.assertNotEqual(summarize([1.,1.]),summarize([.2,1.8]))

    def test_invalid_weights_and_ids_are_rejected(self):
        counts,edges=fixture()
        et=next(iter(edges))
        for bad in [-1.,np.nan,np.inf]:
            changed={key:{k:v.copy() for k,v in value.items()} for key,value in edges.items()}
            changed[et]['edge_attr'][0,0]=bad
            with self.assertRaises(ValueError):extract_sife44(counts,changed,'CA',np.arange(4))
        for ids in [np.array([1,0]),np.array([1,1]),np.array([-1]),np.array([4]),np.array([.5])]:
            with self.assertRaises(ValueError):extract_sife44(counts,edges,'CA',ids)

    def test_matches_reviewed_reference_on_all_valid_fixture_columns(self):
        golden=json.loads((ROOT/'assets/reviewed_fixture_values.json').read_text())
        counts,edges=fixture()
        for nt in counts:
            expected=golden['node_types'][nt]
            actual=extract_sife44(counts,edges,nt,np.arange(4),2)
            np.testing.assert_allclose(actual['sife44'],expected['sife44'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(actual['activity'],expected['activity'],rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
