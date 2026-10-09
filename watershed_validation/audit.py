import json
import numpy as np
import torch
import joblib
from util import BASE,P,save,score
from model import Network
from run import PREP,OUT,predict,rf_features


def main():
    device=torch.device('mps' if torch.backends.mps.is_available() else'cpu');torch.set_num_threads(4);checked=0
    for split in P['spatial_seeds']:
        path=PREP/f'split_{split}'/'development.npy';X=rf_features(path)
        for seed in P['training_seeds']:
            previous=[]
            for budget in [0]+P['target_budgets']:
                common=None
                for architecture in P['architectures']:
                    folder=OUT/f'split_{split}'/f'seed_{seed}'/architecture/f'budget_{budget}'
                    ids=json.loads((folder/'target_ids.json').read_text())
                    assert len(ids)==budget and len(set(ids))==budget and ids[:len(previous)]==previous
                    if common is None:common=ids
                    assert ids==common
                    if architecture=='random_forest':probabilities=joblib.load(folder/'model.joblib').predict_proba(X)
                    else:
                        model=Network(architecture).to(device)
                        model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True))
                        probabilities=predict(model,path,device)
                    blob=np.load(folder/'development.npz',allow_pickle=False)
                    np.testing.assert_allclose(probabilities,blob['probabilities'],atol=1e-5)
                    result=json.loads((folder/'development.json').read_text());recomputed=score(blob['targets'],probabilities,blob['weights'])
                    for key in ['macro_f1','mean_iou','accuracy']:assert np.isclose(result[key],recomputed[key],atol=1e-6)
                    assert result['target_training_labels']==budget and result['target_selection_labels']==0
                    checked+=1
                previous=common
    assert checked==240
    save(OUT/'development_audit.json',{'complete':True,'results_checked':checked,'nested_ids_matched':True,
        'probabilities_recomputed':True,'true_target_training_budget_verified':True,'test_evaluated':False})
    print('All240 development results verified.',flush=True)


if __name__=='__main__':main()
