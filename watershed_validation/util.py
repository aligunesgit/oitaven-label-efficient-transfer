import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
P=json.loads((BASE/'protocol.json').read_text())


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def score(y,p,w):
    assert p.shape==(len(y),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1);support=np.bincount(y,minlength=7);present=np.flatnonzero(support>0)
    cm=confusion_matrix(y,pred,labels=np.arange(7),sample_weight=w)
    denom=cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou=np.divide(np.diag(cm),denom,out=np.zeros(7),where=denom>0)
    recall=recall_score(y,pred,labels=np.arange(7),average=None,sample_weight=w,zero_division=0)
    return {'macro_f1':float(f1_score(y,pred,labels=np.arange(7),average='macro',sample_weight=w,zero_division=0)),
        'mean_iou':float(iou.mean()),'accuracy':float(accuracy_score(y,pred,sample_weight=w)),
        'supported_macro_f1':float(f1_score(y,pred,labels=present,average='macro',sample_weight=w,zero_division=0)),
        'supported_mean_iou':float(iou[present].mean()),'balanced_accuracy':float(recall[present].mean()),
        'support':support.tolist(),'supported_classes':present.tolist(),
        'per_class_recall':[float(recall[c]) if support[c] else None for c in range(7)],
        'per_class_iou':[float(iou[c]) if support[c] else None for c in range(7)]}
