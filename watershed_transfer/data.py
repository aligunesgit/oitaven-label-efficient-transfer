import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
P=json.loads((BASE/'protocol.json').read_text())
PREP=ROOT/'watershed_experiments/prepared'
SETS=['source_train','source_control','target_pool','target_controller','target_development']


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def build():
    out=BASE/'patches';out.mkdir(exist_ok=True)
    manifest_path=out/'manifest.json'
    signatures={name:hashlib.sha256((PREP/f'{name}.npz').read_bytes()).hexdigest() for name in SETS}
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text());assert m['complete'] and m['sample_hashes']==signatures and m['radius']==P['patch_radius']
        return
    radius=P['patch_radius'];assert radius<32
    prepared=json.loads((PREP/'manifest.json').read_text())
    counts={}
    for domain,names in [('Source',['source_train','source_control']),('Target',['target_pool','target_controller','target_development'])]:
        path=ROOT/f'DomainAdaptationDataset/Datasets/{domain}.mat'
        expected=prepared['raw_sha256'][str(path.relative_to(ROOT))]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        image=loadmat(path)[domain]
        for name in names:
            sample=np.load(PREP/f'{name}.npz',allow_pickle=False)
            coords=sample['coords']
            assert ((coords%256>=32)&(coords%256<224)).all()
            array=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=np.float32,
                shape=(len(coords),5,2*radius+1,2*radius+1))
            for i,(row,col) in enumerate(coords):
                patch=image[row-radius:row+radius+1,col-radius:col+radius+1]
                assert patch.shape==(2*radius+1,2*radius+1,5) and np.isfinite(patch).all()
                array[i]=patch.transpose(2,0,1)
            array.flush();counts[name]=len(coords)
            print(name,'patches:',len(coords),flush=True)
        del image
    source=np.load(out/'source_train.npy',mmap_mode='r')
    mean=source.astype(np.float64).mean((0,2,3))
    std=source.astype(np.float64).std((0,2,3));assert (std>0).all()
    np.savez_compressed(out/'normalization.npz',mean=mean,std=std)
    save(manifest_path,{'complete':True,'radius':radius,'sample_hashes':signatures,'counts':counts,
        'normalization_source_only':True,'no_test_patches':True})


def labels(name):
    assert name in ['source_train','source_control','target_controller','target_development']
    return np.load(PREP/f'{name}_labels.npy',allow_pickle=False)


if __name__=='__main__':build()
