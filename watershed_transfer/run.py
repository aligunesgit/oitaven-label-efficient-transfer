import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'matplotlib_cache'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from data import BASE,P,PREP,labels,save
from model import Network,score,gates

OUT=BASE/'results'


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


class Samples(Dataset):
    def __init__(self,name,ids=None,y=None):
        self.maps=np.load(BASE/'patches'/f'{name}.npy',mmap_mode='r')
        self.ids=np.arange(len(self.maps)) if ids is None else np.asarray(ids,dtype=int)
        self.y=y if y is not None else np.zeros(len(self.ids),dtype=np.int64)
        assert len(self.y)==len(self.ids)
        norm=np.load(BASE/'patches/normalization.npz',allow_pickle=False)
        self.mean=norm['mean'].astype(np.float32)[:,None,None]
        self.std=norm['std'].astype(np.float32)[:,None,None]

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        x=(np.array(self.maps[self.ids[index]],copy=True)-self.mean)/self.std
        assert np.isfinite(x).all()
        return torch.from_numpy(x),int(self.y[index])


def predict(model,name,device,ids=None):
    ps=[];zs=[];model.eval()
    with torch.inference_mode():
        for x,_ in DataLoader(Samples(name,ids=ids),batch_size=256,shuffle=False,num_workers=0):
            logits,z=model(x.to(device));ps.append(logits.softmax(1).cpu().numpy());zs.append(z.cpu().numpy())
    probabilities=np.concatenate(ps);features=np.concatenate(zs)
    assert np.isfinite(probabilities).all() and np.isfinite(features).all()
    return probabilities,features


def augment(x,generator):
    flips=torch.randint(0,4,(len(x),),generator=generator)
    x=x.clone()
    for code,dims in [(1,[3]),(2,[2]),(3,[2,3])]:
        mask=flips==code
        x[mask]=torch.flip(x[mask],dims=dims)
    return x


def atomic_torch(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def train(model,source,device,seed,folder,phase,control_y,control_w,
          pseudo=None,prototypes=None,selection=True):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'weights.pt').exists() and (folder/'training.json').exists():
        model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True));return
    settings=P['source_training'] if phase=='source' else P['adaptation']
    maximum,minimum,patience=settings['max_epochs'],settings['min_epochs'],settings['patience']
    if not selection:minimum=maximum;patience=maximum
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=P['source_training']['weight_decay'])
    counts=np.bincount(source.y,minlength=7);assert (counts>0).all()
    weights=(1/counts);weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    best_key=(-1.,-1.);best_state=None;stale=0;history=[];start_epoch=1
    resume=folder/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best_key=tuple(state['best_key']);best_state=state['best_state'];stale=state['stale'];start_epoch=state['epoch']+1
    for epoch in range(start_epoch,maximum+1):
        seed_all(seed+epoch)
        source_loader=DataLoader(source,batch_size=P['source_training']['batch_size'],shuffle=True,num_workers=0,
            generator=torch.Generator().manual_seed(seed+1000+epoch))
        if pseudo is not None and len(pseudo):
            target_loader=DataLoader(pseudo,batch_size=P['source_training']['batch_size'],shuffle=True,num_workers=0,
                generator=torch.Generator().manual_seed(seed+2000+epoch))
            target_iterator=iter(target_loader)
        else:target_iterator=None
        source_generator=torch.Generator().manual_seed(seed+3000+epoch)
        target_generator=torch.Generator().manual_seed(seed+4000+epoch)
        model.train();total=0.;seen=0
        for x,y in source_loader:
            x=augment(x,source_generator).to(device);y=y.to(device)
            optimizer.zero_grad();logits,_=model(x);loss=criterion(logits,y)
            if target_iterator is not None:
                try:tx,ty=next(target_iterator)
                except StopIteration:target_iterator=iter(target_loader);tx,ty=next(target_iterator)
                tx=augment(tx,target_generator).to(device);ty=ty.to(device)
                tlogits,tz=model(tx)
                loss=loss+P['adaptation']['pseudo_loss_weight']*nn.functional.cross_entropy(tlogits,ty)
                if prototypes is not None:
                    losses=[]
                    normalized=nn.functional.normalize(tz,dim=1)
                    for cls in torch.unique(ty):
                        target_mean=nn.functional.normalize(normalized[ty==cls].mean(0),dim=0)
                        losses.append((target_mean-prototypes[int(cls)]).square().sum())
                    loss=loss+P['adaptation']['prototype_loss_weight']*torch.stack(losses).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(y);seen+=len(y)
        record={'epoch':epoch,'loss':total/seen}
        if selection:
            cp,_=predict(model,'target_controller',device)
            cm=score(control_y,cp,control_w);key=(cm['macro_f1'],cm['mean_iou'])
            record.update(control_macro_f1=key[0],control_mean_iou=key[1])
        else:key=(float(epoch),0.)
        if key>best_key:
            best_key=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        history.append(record)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        atomic_torch(resume,{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'best_state':best_state,'best_key':list(best_key),
            'history':history,'stale':stale,'epoch':epoch})
        print(f'{folder.relative_to(OUT)} epoch{epoch}: loss={record["loss"]:.4f}'+
              (f', controlF1={key[0]:.4f}' if selection else ''),flush=True)
        if epoch>=minimum and stale>=patience:break
    assert best_state is not None
    model.load_state_dict(best_state)
    atomic_torch(folder/'weights.pt',best_state)
    save(folder/'training.json',{'epochs':len(history),'checkpoint_key':list(best_key),'selection_target_labels':700 if selection else 0,
         'fixed_final_checkpoint':not selection,'pseudo_samples':len(pseudo) if pseudo is not None else 0})
    if resume.exists():resume.unlink()


def evaluate(seed,name,model,device):
    folder=OUT/f'seed_{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
    probability,_=predict(model,'target_development',device)
    y=labels('target_development');w=np.load(PREP/'target_development.npz',allow_pickle=False)['inclusion_weights']
    result={'seed':seed,'method':name,'target_labels_available':700,'test_evaluated':False,**score(y,probability,w)}
    np.savez_compressed(folder/'development.npz',probabilities=probability,targets=y,weights=w)
    save(folder/'result.json',result);report()


def rf_features(name):
    maps=np.load(BASE/'patches'/f'{name}.npy',mmap_mode='r')
    result=[]
    for start in range(0,len(maps),256):
        x=np.array(maps[start:start+256]);r=x.shape[2]//2
        result.append(np.c_[x[:,:,r,r],x.mean((2,3)),x.std((2,3))])
    return np.concatenate(result)


def report():
    rows=[]
    for file in OUT.glob('seed_*/*/result.json'):
        r=json.loads(file.read_text());rows.append({k:v for k,v in r.items() if not k.startswith('per_class')})
    if not rows:return
    df=pd.DataFrame(rows).sort_values(['seed','method']);df.to_csv(OUT/'metrics.csv',index=False)
    summary=df.groupby('method')[['macro_f1','mean_iou','accuracy','balanced_accuracy']].agg(['mean','std','count'])
    summary.to_csv(OUT/'summary.csv')
    lines=['# Spectral–spatial transfer pilot','',
        'No active learning or risk weights. Argmax7-class decisions. Same source/target spatial partition; source-only patch normalization. Five seeds; target700 labels inform CNN checkpoint selection and class reliability. Direct-training references use these700 labels for fitting; they choose a fixed final checkpoint.2000 development labels are separate evaluation overhead. Reserved test is not evaluated.','',
        'Prototype and self-training implementations are experimental baselines, not exact reproductions or verified novelty. Wilson bounds are heuristic gating signals, not coverage guarantees under spatial dependence. Source-continued control separates additional optimization from adaptation.','',
        '| Method | Seeds | Macro-F1 | mIoU | Accuracy | Balanced accuracy |','|---|---:|---:|---:|---:|---:|']
    for method,row in summary.iterrows():
        lines.append(f'| {method} | {int(row[("macro_f1","count")])} | '+
            ' | '.join(f'{row[(m,"mean")]:.4f}' for m in ['macro_f1','mean_iou','accuracy','balanced_accuracy'])+' |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    OUT.mkdir(exist_ok=True)
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
    frozen['protocol.json']=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest()
    file=OUT/'source_hashes.json'
    if file.exists():assert json.loads(file.read_text())==frozen,'Code changed; use new output directory.'
    else:save(file,frozen)
    source_y=labels('source_train');control_y=labels('target_controller')
    assert set(source_y)==set(range(7)) and set(control_y)==set(range(7))
    control_w=np.load(PREP/'target_controller.npz',allow_pickle=False)['inclusion_weights']
    source=Samples('source_train',y=source_y)
    Xs,Xc,Xd=[rf_features(name) for name in ['source_train','target_controller','target_development']]
    for seed in P['seeds']:
        for name,xt,yt in [('rf_source',Xs,source_y),('rf_target700',np.r_[Xs,Xc],np.r_[source_y,control_y])]:
            folder=OUT/f'seed_{seed}'/name
            if (folder/'result.json').exists():continue
            save(BASE/'status.json',{'state':'running','seed':seed,'method':name,'device':str(device)})
            model=RandomForestClassifier(random_state=seed,**P['rf']);model.fit(xt,yt)
            assert np.array_equal(model.classes_,np.arange(7))
            folder.mkdir(parents=True,exist_ok=True)
            import joblib
            joblib.dump(model,folder/'model.joblib')
            probability=model.predict_proba(Xd);y=labels('target_development');w=np.load(PREP/'target_development.npz',allow_pickle=False)['inclusion_weights']
            np.savez_compressed(folder/'development.npz',probabilities=probability,targets=y,weights=w)
            save(folder/'result.json',{'seed':seed,'method':name,'target_labels_available':0 if name=='rf_source' else 700,
                 'test_evaluated':False,**score(y,probability,w)});report()
        seed_all(seed);teacher=Network().to(device)
        folder=OUT/f'seed_{seed}'/'cnn_source'
        save(BASE/'status.json',{'state':'running','seed':seed,'method':'cnn_source','device':str(device)})
        train(teacher,source,device,seed,folder,'source',control_y,control_w)
        evaluate(seed,'cnn_source',teacher,device)
        initial={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
        source_z=predict(teacher,'source_train',device)[1]
        control_p=predict(teacher,'target_controller',device)[0]
        pool_p,pool_z=predict(teacher,'target_pool',device)
        choices,pseudo_y,prototypes,reliability=gates(source_z,source_y,control_p,control_y,pool_p,pool_z,P)
        save(OUT/f'seed_{seed}'/'pseudo_selection.json',{'all_ids':choices['all'].tolist(),
            'filtered_ids':choices['filtered'].tolist(),'pseudo_labels':pseudo_y.tolist(),'reliability':reliability,
            'unacquired_target_labels_used':False})
        for name in P['methods']:
            if name.startswith('rf') or name=='cnn_source':continue
            folder=OUT/f'seed_{seed}'/name
            if (folder/'result.json').exists():continue
            save(BASE/'status.json',{'state':'running','seed':seed,'method':name,'device':str(device)})
            seed_all(seed);model=Network().to(device);model.load_state_dict(initial)
            pseudo=None;proto=None;dataset=source;selection=True
            if name=='cnn_target700':
                # Mix two datasets without introducing test/development samples.
                from torch.utils.data import ConcatDataset
                dataset=ConcatDataset([source,Samples('target_controller',y=control_y)])
                dataset.y=np.r_[source_y,control_y];selection=False
            elif name!='cnn_source_continued':
                kind='filtered' if name in ['cnn_filtered_self_training','cnn_selective_alignment'] else'all'
                ids=choices[kind];pseudo=Samples('target_pool',ids=ids,y=pseudo_y[ids])
                if name in ['cnn_prototype_all','cnn_selective_alignment']:
                    proto=torch.tensor(prototypes,dtype=torch.float32,device=device)
            train(model,dataset,device,seed+10000,folder,'adaptation',control_y,control_w,pseudo,proto,selection)
            evaluate(seed,name,model,device)
    save(BASE/'status.json',{'state':'complete','test_evaluated':False})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
