"""Independent combinatorial checks for matching and multiple-test adjustment."""
import itertools
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from analyze import optimal_pairs,holm

def exhaustive(positive,negative,caliper):
    d=cdist(positive,negative);best=(0,0.)
    for count in range(1,min(len(positive),len(negative))+1):
        for rows in itertools.combinations(range(len(positive)),count):
            for cols in itertools.permutations(range(len(negative)),count):
                vals=d[list(rows),list(cols)]
                if np.all(vals<=caliper):
                    if count>best[0] or (count==best[0] and vals.sum()<best[1]):best=(count,vals.sum())
    return best

class AnalysisChecks(unittest.TestCase):
    def test_assignment_against_exhaustive_search(self):
        rng=np.random.default_rng(511)
        for _ in range(40):
            p=rng.uniform(0,1,(3,2));n=rng.uniform(0,1,(4,2))
            pairs,dist=optimal_pairs(p,n,.5);count,total=exhaustive(p,n,.5)
            self.assertEqual(len(pairs),count);self.assertAlmostEqual(float(dist.sum()),float(total),places=12)
            self.assertEqual(len(set(pairs[:,0])),len(pairs));self.assertEqual(len(set(pairs[:,1])),len(pairs))
    def test_empty_support(self):
        pairs,dist=optimal_pairs(np.ones((2,2)),np.ones((0,2)),.5)
        self.assertEqual(pairs.shape,(0,2));self.assertEqual(len(dist),0)
    def test_no_overlap(self):
        pairs,_=optimal_pairs(np.zeros((3,2)),np.ones((4,2))*10,.5)
        self.assertEqual(len(pairs),0)
    def test_zero_caliper_and_deterministic_duplicate_values(self):
        p=np.zeros((3,2));n=np.zeros((4,2))
        first,d=optimal_pairs(p,n,0)
        self.assertEqual(len(first),3);self.assertTrue(np.all(d==0))
        np.testing.assert_array_equal(first,optimal_pairs(p,n,0)[0])
    def test_holm_known_values(self):
        np.testing.assert_allclose(holm([.01,.04,.03,.002]),[.03,.06,.06,.008],rtol=0,atol=1e-12)

if __name__=='__main__':unittest.main()
