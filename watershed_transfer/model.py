import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.spatial=nn.Sequential(nn.Conv2d(5,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.embedding=nn.Sequential(nn.Linear(64,64),nn.LayerNorm(64),nn.ReLU())
        self.head=nn.Linear(64,7)

    def forward(self,x):
        center=x[:,:,x.shape[2]//2,x.shape[3]//2]
        z=self.embedding(torch.cat([self.spectral(center),self.spatial(x).flatten(1)],1))
        return self.head(z),z


def score(y,p,weights):
    assert p.shape==(len(y),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1)
    matrix=confusion_matrix(y,pred,labels=np.arange(7),sample_weight=weights)
    denominator=matrix.sum(0)+matrix.sum(1)-np.diag(matrix)
    iou=np.divide(np.diag(matrix),denominator,out=np.zeros(7),where=denominator>0)
    recall=recall_score(y,pred,labels=np.arange(7),average=None,sample_weight=weights,zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred,sample_weight=weights)),
        'macro_f1':float(f1_score(y,pred,labels=np.arange(7),average='macro',sample_weight=weights,zero_division=0)),
        'mean_iou':float(iou.mean()),'balanced_accuracy':float(recall.mean()),
        'per_class_iou':iou.tolist(),'per_class_recall':recall.tolist()}


def wilson_lower(correct,count,z=1.96):
    if count==0:return 0.
    p=correct/count
    return (p+z*z/(2*count)-z*np.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count)


def gates(source_z,source_y,control_p,control_y,pool_p,pool_z,settings):
    normalized=lambda z:z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    sz,pz=normalized(source_z),normalized(pool_z)
    prototypes=np.stack([sz[source_y==c].mean(0) for c in range(7)])
    prototypes=normalized(prototypes)
    pred=control_p.argmax(1);pool_pred=pool_p.argmax(1);confidence=pool_p.max(1)
    rows=[];eligible=np.zeros(7,dtype=bool);similarity_limits=[]
    for c in range(7):
        mask=pred==c;count=int(mask.sum());correct=int((control_y[mask]==c).sum())
        lower=wilson_lower(correct,count,settings['class_reliability']['z'])
        threshold=float(np.quantile(np.einsum('ij,j->i',sz[source_y==c],prototypes[c]),
            settings['pseudo_labels']['prototype_similarity_quantile']))
        eligible[c]=count>=settings['class_reliability']['minimum_predictions'] and lower>=settings['class_reliability']['wilson_precision_lower_bound']
        similarity_limits.append(threshold)
        rows.append({'class':c,'predictions':count,'correct':correct,'wilson_lower':lower,
                     'eligible':bool(eligible[c]),'source_similarity_threshold':threshold})
    similarity=np.einsum('ij,ij->i',pz,prototypes[pool_pred])
    generic=confidence>=settings['pseudo_labels']['confidence']
    filtered=generic & eligible[pool_pred] & (similarity>=np.asarray(similarity_limits)[pool_pred])
    choices={}
    for name,mask in [('all',generic),('filtered',filtered)]:
        ids=[]
        for c in range(7):
            candidates=np.flatnonzero(mask & (pool_pred==c))
            if len(candidates)<settings['pseudo_labels']['min_class_candidates']:continue
            ordered=candidates[np.argsort(-confidence[candidates],kind='stable')]
            ids.extend(ordered[:settings['pseudo_labels']['cap_per_class']].tolist())
        choices[name]=np.asarray(ids,dtype=int)
    return choices,pool_pred,prototypes,rows
