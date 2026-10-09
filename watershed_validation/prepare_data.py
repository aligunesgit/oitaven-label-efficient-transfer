import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.io import loadmat
from util import BASE,ROOT,P,save

# Reuse previously verified block assignment and sampling code; originals are left unchanged.
sys.path.insert(1,str(ROOT/'watershed_experiments'))
import prepare as legacy
from common import split_centers


def patches(image,coords,path):
    radius=P['patch_radius']
    assert ((coords%256>=32)&(coords%256<224)).all()
    array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(coords),5,15,15))
    for i,(r,c) in enumerate(coords):
        value=image[r-radius:r+radius+1,c-radius:c+radius+1]
        assert value.shape==(15,15,5) and np.isfinite(value).all()
        array[i]=value.transpose(2,0,1)
    array.flush()


def main():
    out=BASE/'prepared';out.mkdir(exist_ok=True)
    original=ROOT/'watershed_experiments/prepared'
    signatures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'protocol.json',BASE/'prepare_data.py',ROOT/'watershed_experiments/prepare.py',ROOT/'watershed_experiments/common.py']}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text());assert old['complete'] and old['signatures']==signatures;return
    for name in ['source_train','source_control']:
        destination=out/f'{name}.npy'
        if not destination.exists():os.link(ROOT/f'watershed_transfer/patches/{name}.npy',destination)
        shutil.copyfile(original/f'{name}_labels.npy',out/f'{name}_labels.npy')
    shutil.copyfile(ROOT/'watershed_transfer/patches/normalization.npz',out/'normalization.npz')
    frozen=ROOT/'watershed_experiments/diagnostics/rejected_geometry_split/prepared/Target_block_assignments.json'
    assignment=json.loads(frozen.read_text());test_blocks={int(b) for b,g in assignment.items() if g==3}
    raw=ROOT/'DomainAdaptationDataset/Datasets/Target.mat'
    expected=json.loads((original/'manifest.json').read_text())['raw_sha256'][str(raw.relative_to(ROOT))]
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==expected
    image=loadmat(raw)['Target'];gt=loadmat(ROOT/'DomainAdaptationDataset/Datasets/Target_gt.mat')['Target_gt']
    coords,blocks,_,_=split_centers(gt.shape,8,[1.],P['spatial_seeds'][0])
    parent=coords[:,0]//512*((gt.shape[1]+511)//512)+coords[:,1]//512
    band=image[coords[:,0],coords[:,1]]
    valid=(gt[coords[:,0],coords[:,1]]>0)&np.isfinite(band).all(1)&(band!=0).any(1)&~np.isin(parent,list(test_blocks))
    coords,blocks=coords[valid],blocks[valid]
    y=gt[coords[:,0],coords[:,1]].astype(int)-1
    partitions={}
    for seed in P['spatial_seeds']:
        folder=out/f'split_{seed}';folder.mkdir(exist_ok=True)
        group,lookup=legacy.assign_non_test_blocks(blocks,y,seed)
        save(folder/'block_assignments.json',lookup)
        rng=np.random.default_rng(seed)
        legacy.OUT=folder
        for group_id,name,cap in [(1,'target_reference',P['target_reference_cap']),(2,'development',P['development_cap'])]:
            minimum=P['minimum_reference_per_class'] if group_id==1 else P['minimum_development_per_class']
            legacy.write(name,image,gt,coords[group==group_id],blocks[group==group_id],cap,rng,minimum_per_class=minimum)
            sampled=np.load(folder/f'{name}.npz',allow_pickle=False)
            assert not set(sampled['parent_blocks'])&test_blocks
            patches(image,sampled['coords'],folder/f'{name}.npy')
        reference=np.load(folder/'target_reference.npz',allow_pickle=False)
        development=np.load(folder/'development.npz',allow_pickle=False)
        assert not set(reference['blocks'])&set(development['blocks'])
        partitions[str(seed)]={'reference':len(reference['coords']),'development':len(development['coords']),
            'reference_blocks':np.unique(reference['blocks']).tolist(),'development_blocks':np.unique(development['blocks']).tolist()}
        print('Prepared spatial split',seed,flush=True)
    save(out/'manifest.json',{'complete':True,'signatures':signatures,'partitions':partitions,
        'reserved_test_blocks':sorted(test_blocks),'test_labels_not_read_for_partitioning':True,
        'source_only_normalization':True,'benchmark_non_test_labels_used_for_class_support':True})


if __name__=='__main__':main()
