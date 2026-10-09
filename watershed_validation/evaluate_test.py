"""One final batch on fixed test samples after every model has been frozen."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.io import loadmat
from util import BASE,ROOT,P,save,score
from model import Network
from run import OUT,predict,rf_features


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=json.loads((OUT/'development_audit.json').read_text());assert audit['complete'] and audit['results_checked']==240
    plans=[]
    for seed in P['training_seeds']:
        for architecture in P['architectures']:
            folder=OUT/'source_models'/f'seed_{seed}'/architecture
            file=folder/('model.joblib' if architecture=='random_forest' else'weights.pt')
            plans.append({'split':None,'seed':seed,'architecture':architecture,'budget':0,
                          'folder':str(folder.relative_to(OUT)),'checkpoint':str(file.relative_to(OUT)),'sha256':digest(file)})
    for split in P['spatial_seeds']:
        for seed in P['training_seeds']:
            for architecture in P['architectures']:
                for budget in P['target_budgets']:
                    folder=OUT/f'split_{split}'/f'seed_{seed}'/architecture/f'budget_{budget}'
                    file=folder/('model.joblib' if architecture=='random_forest' else'weights.pt')
                    plans.append({'split':split,'seed':seed,'architecture':architecture,'budget':budget,
                        'folder':str(folder.relative_to(OUT)),'checkpoint':str(file.relative_to(OUT)),'sha256':digest(file)})
    test_dir=OUT/'final_test';test_dir.mkdir(exist_ok=True)
    raw_test=ROOT/'watershed_experiments/prepared'
    plan={'protocol_sha256':digest(BASE/'protocol.json'),'models':plans,
          'test_sample_sha256':digest(raw_test/'target_test.npz'),'test_label_sha256':digest(raw_test/'target_test_labels.npy')}
    freeze=test_dir/'evaluation_plan.json'
    if freeze.exists():assert json.loads(freeze.read_text())==plan,'Models changed after test access; abort.'
    else:save(freeze,plan)
    if (test_dir/'audit.json').exists():
        assert json.loads((test_dir/'audit.json').read_text())['complete'];print('Final test batch already complete; retained.');return
    # No label values are read until the complete checkpoint plan above exists.
    coords=np.load(raw_test/'target_test.npz',allow_pickle=False)['coords']
    radius=P['patch_radius'];assert ((coords%512>=64)&(coords%512<448)).all()
    patches=test_dir/'test_patches.npy'
    if not patches.exists():
        image=loadmat(ROOT/'DomainAdaptationDataset/Datasets/Target.mat')['Target']
        array=np.lib.format.open_memmap(patches,mode='w+',dtype=np.float32,shape=(len(coords),5,15,15))
        for i,(r,c) in enumerate(coords):
            patch=image[r-radius:r+radius+1,c-radius:c+radius+1];assert np.isfinite(patch).all()
            array[i]=patch.transpose(2,0,1)
        array.flush();del image
    y=np.load(raw_test/'target_test_labels.npy',allow_pickle=False);weights=np.ones(len(y))
    assert len(y)==len(coords)
    save(test_dir/'class_support.json',{'counts':np.bincount(y,minlength=7).tolist(),
        'all_seven_classes_represented':len(np.unique(y))==7,
        'interpretation':'Absent classes unsupported; do not infer their recall or seven-class superiority.'})
    device=torch.device('mps' if torch.backends.mps.is_available() else'cpu');torch.set_num_threads(4);X=rf_features(patches)
    rows=[]
    for item in plans:
        file=OUT/item['checkpoint'];assert digest(file)==item['sha256']
        name=item['folder'].replace('/','__');saved=test_dir/f'{name}.json'
        if saved.exists():r=json.loads(saved.read_text())
        else:
            if item['architecture']=='random_forest':probabilities=joblib.load(file).predict_proba(X)
            else:
                model=Network(item['architecture']).to(device)
                model.load_state_dict(torch.load(file,map_location=device,weights_only=True));probabilities=predict(model,patches,device)
            r={k:item[k] for k in ['split','seed','architecture','budget']};r.update(score(y,probabilities,weights))
            np.savez_compressed(test_dir/f'{name}.npz',probabilities=probabilities,targets=y)
            save(saved,r)
        rows.append({k:v for k,v in r.items() if k not in ['support','supported_classes','per_class_recall','per_class_iou']})
        pd.DataFrame(rows).to_csv(test_dir/'metrics.csv',index=False)
        print('Final test:',name,flush=True)
    assert len(rows)==200
    pd.DataFrame(rows).groupby(['architecture','budget'])[['macro_f1','mean_iou','accuracy','supported_macro_f1','supported_mean_iou']].agg(['mean','std','count']).to_csv(test_dir/'summary.csv')
    save(test_dir/'audit.json',{'complete':True,'fixed_models_evaluated':len(rows),'test_evaluated':True,
        'models_frozen_before_test_label_access':True,'no_fitting_after_test_access':True})


if __name__=='__main__':main()
