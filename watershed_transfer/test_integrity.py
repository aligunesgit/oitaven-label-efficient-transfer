import unittest
import numpy as np
import torch
from data import P
from model import Network,score,wilson_lower,gates


class IntegrityTests(unittest.TestCase):
    def test_network_trains_and_outputs_seven_classes(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        model=Network();x=torch.randn(14,5,15,15);y=torch.arange(14)%7
        before=model.spatial[0].weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        logits,z=model(x);self.assertEqual(tuple(logits.shape),(14,7));self.assertEqual(tuple(z.shape),(14,64))
        loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();opt.step()
        self.assertGreater(float((model.spatial[0].weight.detach()-before).abs().max()),0.)

    def test_perfect_scores_and_uncertain_class_bound(self):
        y=np.arange(7);p=np.eye(7)
        s=score(y,p,np.ones(7))
        self.assertEqual(s['macro_f1'],1.);self.assertEqual(s['mean_iou'],1.)
        self.assertEqual(wilson_lower(0,0),0.)
        self.assertLess(wilson_lower(1,1),.6)
        self.assertGreater(wilson_lower(30,30),.6)

    def test_unreliable_class_is_removed_without_pool_truth(self):
        y=np.repeat(np.arange(7),30);z=np.eye(7)[y]
        p=np.full((len(y),7),.001);p[np.arange(len(y)),y]=.994
        control_y=y.copy();control_y[:30]=1
        choices,pred,proto,rows=gates(z,y,p,control_y,p,z,P)
        self.assertEqual(len(choices['all']),210)
        self.assertFalse(np.isin(choices['filtered'],np.arange(30)).any())
        self.assertTrue(set(choices['filtered'])<=set(choices['all']))
        self.assertFalse(rows[0]['eligible'])
        self.assertEqual(len(np.unique(choices['filtered'])),len(choices['filtered']))


if __name__=='__main__':unittest.main()
