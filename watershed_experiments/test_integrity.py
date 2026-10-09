import unittest
import numpy as np
from common import split_centers, P, choose, cost_matrix, adaptive_weights, coral


class IntegrityTests(unittest.TestCase):
    def test_spatial_blocks_and_patches_do_not_overlap(self):
        coords, blocks, groups, assignments = split_centers((1800,1800),8,[.6,.15,.1,.15],9)
        for i in range(4):
            for j in range(i+1,4):
                self.assertFalse(set(blocks[groups==i]) & set(blocks[groups==j]))
        self.assertTrue(((coords%P['block_pixels'] >= P['buffer_pixels']) &
                         (coords%P['block_pixels'] < P['block_pixels']-P['buffer_pixels'])).all())
        self.assertLess(P['patch_radius'],P['buffer_pixels'])

    def test_query_unique_and_independent_of_hidden_labels(self):
        rng=np.random.default_rng(3)
        p=rng.dirichlet(np.ones(7),size=30)
        available=np.arange(10,40)
        for method in ['fixed_random','fixed_entropy','fixed_risk3','adaptive_entropy','adaptive_risk','budgetmatched_risk']:
            chosen=choose(method,p,available,12,np.ones(7),np.random.default_rng(3))
            self.assertEqual(len(set(chosen)),12)
            self.assertTrue(set(chosen)<=set(available))

    def test_risk_matrix_orientation_and_rare_support(self):
        matrix=cost_matrix()
        self.assertEqual(matrix[0,6],3)
        self.assertEqual(matrix[6,0],1)
        np.testing.assert_array_equal(np.diag(matrix),np.zeros(7))
        y=np.array([0]*20+[1]*4)
        p=np.zeros((24,7));p[:,6]=1
        weights,support,rates=adaptive_weights(y,p)
        self.assertEqual(weights[0],7)
        self.assertEqual(weights[1],1)
        self.assertTrue(np.isfinite(rates).all())

    def test_coral_identity_when_domains_match(self):
        X=np.random.default_rng(3).normal(size=(50,15))
        adapted,*_=coral(X,X)
        np.testing.assert_allclose(adapted,X,atol=1e-8)


if __name__=='__main__':
    unittest.main()
