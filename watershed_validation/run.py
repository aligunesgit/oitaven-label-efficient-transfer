import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
import joblib
from util import BASE,ROOT,P,save,score
from model import Network

PREP=BASE/'prepared';OUT=BASE/'results'


class Samples(Dataset):
    def __init__(self,path,y=None,ids=None):
        self.maps=np.load(path,mmap_mode='r');self.ids=np.arange(len(self.maps)) if ids is None else np.asarray(ids,dtype=int)
        self.y=np.zeros(len(self.ids),dtype=np.int64) if y is None else np.asarray(y,dtype=np.int64)
        assert len(self.y)==len(self.ids)
        norm=np.load(PREP/'normalization.npz',allow_pickle=False)
        self.mean=norm['mean'].astype(np.float32)[:,None,None];self.std=norm['std'].astype(np.float32)[:,None,None]

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        x=(np.array(self.maps[self.ids[index]],copy=True)-self.mean)/self.std
        assert np.isfinite(x).all();return torch.from_numpy(x),int(self.y[index])


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def atomic_torch(path,data):
    temporary=path.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(path)


def predict(model,path,device):
    ps=[];model.eval()
    with torch.inference_mode():
        for x,_ in DataLoader(Samples(path),batch_size=256,shuffle=False,num_workers=0):
            ps.append(model(x.to(device)).softmax(1).cpu().numpy())
    p=np.concatenate(ps);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return p


def fit_cnn(model,dataset,epochs,lr,seed,folder,device):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'weights.pt').exists() and (folder/'training.json').exists():
        model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True));return
    y=dataset.y;counts=np.bincount(y,minlength=7);assert (counts>0).all()
    weights=1/counts;weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=P['weight_decay'])
    history=[];start_epoch=1;resume=folder/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];start_epoch=state['epoch']+1
    for epoch in range(start_epoch,epochs+1):
        seed_all(seed+epoch)
        loader=DataLoader(dataset,batch_size=P['batch_size'],shuffle=True,num_workers=0,
            generator=torch.Generator().manual_seed(seed+1000+epoch))
        flip_generator=torch.Generator().manual_seed(seed+2000+epoch)
        model.train();loss_sum=0.;seen=0;start=time.time()
        for x,y in loader:
            flips=torch.randint(0,4,(len(x),),generator=flip_generator)
            for code,dims in [(1,[3]),(2,[2]),(3,[2,3])]:
                mask=flips==code;x[mask]=torch.flip(x[mask],dims=dims)
            x=x.to(device);y=y.to(device);optimizer.zero_grad()
            loss=criterion(model(x),y);assert torch.isfinite(loss)
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            optimizer.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
        history.append({'epoch':epoch,'loss':loss_sum/seen,'samples_seen':seen,'updates':len(loader),'seconds':time.time()-start})
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        atomic_torch(resume,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'history':history,'epoch':epoch})
        print(f'{folder.relative_to(OUT)} epoch{epoch}/{epochs}: loss={loss_sum/seen:.4f}',flush=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values())
    atomic_torch(folder/'weights.pt',state)
    save(folder/'training.json',{'epochs':epochs,'samples':len(dataset),'target_early_stopping_labels':0,
        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':'fixed final epoch'})
    if resume.exists():resume.unlink()


def rf_features(path):
    maps=np.load(path,mmap_mode='r');parts=[]
    for start in range(0,len(maps),256):
        x=np.array(maps[start:start+256]);r=x.shape[2]//2
        parts.append(np.c_[x[:,:,r,r],x.mean((2,3)),x.std((2,3))])
    return np.concatenate(parts)


def summarize():
    rows=[]
    for path in OUT.glob('split_*/seed_*/*/budget_*/development.json'):
        r=json.loads(path.read_text());rows.append({k:v for k,v in r.items() if not k.startswith('per_class') and k not in ['support','supported_classes']})
    if not rows:return
    df=pd.DataFrame(rows).sort_values(['split','seed','budget','architecture']);df.to_csv(OUT/'development_metrics.csv',index=False)
    summary=df.groupby(['architecture','budget'])[['macro_f1','mean_iou','accuracy']].agg(['mean','std','count'])
    summary.to_csv(OUT/'development_summary.csv')
    lines=['# Architecture, budget and spatial-partition validation','',
        'Three spatial partitions × five training seeds × three budgets × four architectures. Source pretraining40 epochs; target fine-tuning30 epochs. No target validation/early-stopping labels. Budgets100/300/700 are the total target training labels.2000 development references per partition are evaluation overhead. Nested sample IDs are matched across architectures. Same source normalization and patch footprint.','',
        'patch_cnn removes the explicit center-spectrum branch but still sees all five spectral bands. Parameter counts are recorded; architectures are not capacity matched. Non-test benchmark labels enforce class-supported partitions; this is not a label-free deployment sampler.','',
        'Spatial partitions reuse one scene and can overlap across repetitions; do not treat the15 split/seed combinations as15 independent scenes. Reserved test is read only after all models are frozen.','',
        '| Architecture | Target labels | Runs | Macro-F1 | mIoU | Accuracy |','|---|---:|---:|---:|---:|---:|']
    for (architecture,budget),r in summary.iterrows():
        lines.append(f'| {architecture} | {budget} | {int(r[("macro_f1","count")])} | '+
            ' | '.join(f'{r[(m,"mean")]:.4f}' for m in ['macro_f1','mean_iou','accuracy'])+' |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else'cpu')
    OUT.mkdir(exist_ok=True)
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
    frozen['protocol.json']=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    freeze=OUT/'source_hashes.json'
    if freeze.exists():assert json.loads(freeze.read_text())==frozen,'Source changed; use a new study folder.'
    else:save(freeze,frozen)
    source_y=np.load(PREP/'source_train_labels.npy',allow_pickle=False)
    source=Samples(PREP/'source_train.npy',y=source_y);Xs=rf_features(PREP/'source_train.npy')
    for seed in P['training_seeds']:
        for kind in P['architectures']:
            folder=OUT/'source_models'/f'seed_{seed}'/kind
            if kind=='random_forest':
                folder.mkdir(parents=True,exist_ok=True)
                if not (folder/'model.joblib').exists():
                    model=RandomForestClassifier(random_state=seed,**P['rf']);model.fit(Xs,source_y);joblib.dump(model,folder/'model.joblib')
            else:
                save(BASE/'status.json',{'state':'running','stage':'source','seed':seed,'architecture':kind,'device':str(device)})
                seed_all(seed);model=Network(kind).to(device)
                fit_cnn(model,source,P['source_epochs'],P['source_lr'],seed,folder,device)
    for split in P['spatial_seeds']:
        data=PREP/f'split_{split}'
        reference_y=np.load(data/'target_reference_labels.npy',allow_pickle=False)
        dev_y=np.load(data/'development_labels.npy',allow_pickle=False)
        dev_w=np.load(data/'development.npz',allow_pickle=False)['inclusion_weights']
        assert (np.bincount(dev_y,minlength=7)>=20).all()
        Xr=rf_features(data/'target_reference.npy');Xd=rf_features(data/'development.npy')
        for seed in P['training_seeds']:
            permutation=np.random.default_rng(seed).permutation(len(reference_y))
            for budget in [0]+P['target_budgets']:
                ids=permutation[:budget]
                for kind in P['architectures']:
                    folder=OUT/f'split_{split}'/f'seed_{seed}'/kind/f'budget_{budget}'
                    if (folder/'development.json').exists():continue
                    save(BASE/'status.json',{'state':'running','stage':'finetune','split':split,'seed':seed,'budget':budget,'architecture':kind,'device':str(device)})
                    folder.mkdir(parents=True,exist_ok=True);save(folder/'target_ids.json',ids.tolist())
                    if kind=='random_forest':
                        if budget==0:
                            model=joblib.load(OUT/'source_models'/f'seed_{seed}'/kind/'model.joblib')
                        else:
                            model=RandomForestClassifier(random_state=seed,**P['rf'])
                            model.fit(np.r_[Xs,Xr[ids]],np.r_[source_y,reference_y[ids]])
                        joblib.dump(model,folder/'model.joblib');probabilities=model.predict_proba(Xd)
                    else:
                        seed_all(seed);model=Network(kind).to(device)
                        model.load_state_dict(torch.load(OUT/'source_models'/f'seed_{seed}'/kind/'weights.pt',map_location=device,weights_only=True))
                        target=Samples(data/'target_reference.npy',y=reference_y[ids],ids=ids)
                        dataset=ConcatDataset([source,target]);dataset.y=np.r_[source_y,reference_y[ids]]
                        if budget:
                            fit_cnn(model,dataset,P['finetune_epochs'],P['finetune_lr'],seed+10000,folder,device)
                        else:
                            atomic_torch(folder/'weights.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
                        probabilities=predict(model,data/'development.npy',device)
                    result={'split':split,'seed':seed,'budget':budget,'architecture':kind,
                        'target_training_labels':budget,'target_selection_labels':0,'test_evaluated':False,**score(dev_y,probabilities,dev_w)}
                    np.savez_compressed(folder/'development.npz',probabilities=probabilities,targets=dev_y,weights=dev_w)
                    save(folder/'development.json',result);summarize()
    summarize();save(BASE/'status.json',{'state':'development_complete','test_evaluated':False})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
