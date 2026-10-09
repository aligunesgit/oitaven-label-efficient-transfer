"""Reconstruct local arrays from original raw data and published frozen sampling metadata."""
import argparse
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
FIXTURES=ROOT/'frozen_sampling'
COMMIT='e17fff566e4f77bf6b4d7cd3fa995e25627c177f'
sys.path.insert(0,str(ROOT/'watershed_experiments'))
import prepare as legacy


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def write_sample(name,metadata,image,gt,out):
    metadata=json.loads(metadata.read_text())
    arrays={k:np.asarray(v) for k,v in metadata.items()}
    coords=arrays['coords'].astype(np.int64)
    arrays['coords']=coords
    features=legacy.extract(image,coords)
    y=gt[coords[:,0],coords[:,1]].astype(np.int64)-1
    assert ((y>=0)&(y<7)).all()
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'{name}.npz',X=features,**arrays)
    np.save(out/f'{name}_labels.npy',y)


def patch_array(image,metadata,path):
    coords=np.asarray(json.loads(metadata.read_text())['coords'],dtype=int)
    path.parent.mkdir(parents=True,exist_ok=True)
    maps=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(coords),5,15,15))
    for i,(r,c) in enumerate(coords):
        value=image[r-7:r+8,c-7:c+8]
        assert value.shape==(15,15,5) and np.isfinite(value).all()
        maps[i]=value.transpose(2,0,1)
    maps.flush()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    dataset=ROOT/'DomainAdaptationDataset'
    if not dataset.exists() and args.download:
        subprocess.run(['git','clone','https://gitlab.citius.gal/HPC4RS/DomainAdaptationDataset.git',str(dataset)],check=True)
        subprocess.run(['git','-C',str(dataset),'checkout',COMMIT],check=True)
    assert dataset.exists(),'Clone DomainAdaptationDataset beside bootstrap.py, or use --download.'
    expected=json.loads((ROOT/'DomainAdaptationDataset_download_check.json').read_text())
    for item in expected['files']:
        if item['file']=='Datasets/Oiteven.mat':continue
        path=dataset/item['file']
        assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],f'Raw-file mismatch: {path.name}'
    base=ROOT/'watershed_experiments/prepared'
    transfer=ROOT/'watershed_transfer/patches'
    validation=ROOT/'watershed_validation/prepared'
    for domain,names in [('Source',['source_train','source_control']),
                          ('Target',['target_pool','target_controller','target_development','target_test'])]:
        image=loadmat(dataset/f'Datasets/{domain}.mat')[domain]
        gt=loadmat(dataset/f'Datasets/{domain}_gt.mat')[domain+'_gt']
        for name in names:
            fixture=FIXTURES/'base'/f'{name}.json'
            write_sample(name,fixture,image,gt,base)
            if name!='target_test':patch_array(image,fixture,transfer/f'{name}.npy')
        if domain=='Source':
            validation.mkdir(parents=True,exist_ok=True)
            for name in names:
                shutil.copyfile(transfer/f'{name}.npy',validation/f'{name}.npy')
                shutil.copyfile(base/f'{name}_labels.npy',validation/f'{name}_labels.npy')
        else:
            for split in [20261008,20261009,20261010]:
                folder=validation/f'split_{split}'
                for name in ['target_reference','development']:
                    fixture=FIXTURES/f'split_{split}'/f'{name}.json'
                    write_sample(name,fixture,image,gt,folder)
                    patch_array(image,fixture,folder/f'{name}.npy')
                shutil.copyfile(FIXTURES/f'split_{split}'/'block_assignments.json',folder/'block_assignments.json')
        del image,gt
    source=np.load(transfer/'source_train.npy',mmap_mode='r')
    mean=source.astype(np.float64).mean((0,2,3));std=source.astype(np.float64).std((0,2,3));assert (std>0).all()
    for folder in [transfer,validation]:np.savez_compressed(folder/'normalization.npz',mean=mean,std=std)
    for name in ['manifest.json','Source_block_assignments.json','Target_block_assignments.json']:
        shutil.copyfile(FIXTURES/'base'/name,base/name)
    original=ROOT/'watershed_experiments/diagnostics/rejected_geometry_split/prepared'
    original.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(FIXTURES/'original_test_Target_block_assignments.json',original/'Target_block_assignments.json')
    shutil.copyfile(FIXTURES/'original_test_manifest.json',original/'manifest.json')
    for name in ['target_test.npz','target_test_labels.npy']:shutil.copyfile(base/name,original/name)
    # Generated-file hashes depend on the local serialization environment; freeze this reconstruction.
    sample_names=['source_train','source_control','target_pool','target_controller','target_development']
    save(transfer/'manifest.json',{'complete':True,'radius':7,
        'sample_hashes':{n:hashlib.sha256((base/f'{n}.npz').read_bytes()).hexdigest() for n in sample_names},
        'counts':{n:len(np.load(base/f'{n}.npz',allow_pickle=False)['coords']) for n in sample_names},
        'normalization_source_only':True,'no_test_patches':True})
    v=ROOT/'watershed_validation'
    signatures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [v/'protocol.json',v/'prepare_data.py',ROOT/'watershed_experiments/prepare.py',ROOT/'watershed_experiments/common.py']}
    save(validation/'manifest.json',{'complete':True,'signatures':signatures,
        'frozen_sampling_reconstruction':True,'source_only_normalization':True})
    print('Frozen samples,features,labels and source normalization reconstructed. Test labels are evaluation-only.',flush=True)


if __name__=='__main__':main()
