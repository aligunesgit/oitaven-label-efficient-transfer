import unittest
import numpy as np
import torch
from model import Network
from util import score


class IntegrityTests(unittest.TestCase):
    def test_architectures_learn(self):
        torch.set_num_threads(2)
        for kind in ['spectral_mlp','patch_cnn','joint_cnn']:
            model=Network(kind);x=torch.randn(14,5,15,15);y=torch.arange(14)%7
            before=model.head.weight.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            logits=model(x);self.assertEqual(tuple(logits.shape),(14,7))
            torch.nn.functional.cross_entropy(logits,y).backward();optimizer.step()
            self.assertGreater(float((model.head.weight.detach()-before).abs().max()),0.)
            self.assertEqual(model.spatial is None,kind=='spectral_mlp')
            self.assertEqual(model.spectral is None,kind=='patch_cnn')

    def test_nested_equal_budget_sampling(self):
        permutation=np.random.default_rng(42).permutation(700)
        self.assertTrue(np.array_equal(permutation[:100],permutation[:300][:100]))
        self.assertEqual(len(set(permutation[:700])),700)

    def test_partial_test_support_is_explicit(self):
        y=np.array([0,6]);p=np.eye(7)[y];result=score(y,p,np.ones(2))
        self.assertEqual(result['supported_macro_f1'],1.)
        self.assertEqual(result['supported_mean_iou'],1.)
        self.assertIsNone(result['per_class_recall'][1])
        self.assertEqual(result['supported_classes'],[0,6])
        self.assertAlmostEqual(result['macro_f1'],2/7)


if __name__=='__main__':unittest.main()
