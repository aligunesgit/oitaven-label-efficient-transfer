import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from data import BASE,P,save
from model import Network,score
from run import predict,rf_features,OUT


def main():
    device=torch.device('mps' if torch.backends.mps.is_available() else'cpu')
    torch.set_num_threads(4)
    X=rf_features('target_development');checked=0
    for seed in P['seeds']:
        for method in P['methods']:
            folder=OUT/f'seed_{seed}'/method
            b=np.load(folder/'development.npz',allow_pickle=False)
            r=json.loads((folder/'result.json').read_text())
            if method.startswith('rf'):
                model=joblib.load(folder/'model.joblib');probabilities=model.predict_proba(X)
            else:
                model=Network().to(device)
                model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True))
                probabilities=predict(model,'target_development',device)[0]
            np.testing.assert_allclose(probabilities,b['probabilities'],atol=1e-5)
            recomputed=score(b['targets'],probabilities,b['weights'])
            for metric in ['macro_f1','mean_iou','accuracy','balanced_accuracy']:
                assert np.isclose(recomputed[metric],r[metric],atol=1e-6)
            checked+=1
    save(OUT/'audit.json',{'complete':True,'models_checked':checked,'probabilities_recomputed':True,
        'metrics_recomputed':True,'active_learning':False,'test_evaluated':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(OUT/'metrics.csv');summary=d.groupby('method')[['macro_f1','mean_iou']].agg(['mean','std'])
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,['macro_f1','mean_iou']):
        y=np.arange(len(summary));ax.barh(y,summary[(metric,'mean')],xerr=summary[(metric,'std')])
        ax.set_yticks(y,summary.index);ax.set_xlabel(metric);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Five-seed development mean ± SD; test not evaluated');fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf')
    print('All',checked,'models independently verified.',flush=True)


if __name__=='__main__':main()
