import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import BASE,P,metrics,save,multiply
from scipy.special import softmax


def main():
    root=BASE/'pilot'
    pool_size=len(np.load(BASE/'prepared/target_pool.npz',allow_pickle=False)['X'])
    transforms=np.load(root/'transforms.npz',allow_pickle=False)
    raw=np.load(BASE/'prepared/target_development.npz',allow_pickle=False)['X']
    development=(raw.astype(float)-transforms['scaler_mean'])/transforms['scaler_scale']
    oracle=np.load(BASE/'prepared/target_pool_labels.npy',allow_pickle=False)
    rows=0
    for seed in P['seeds']:
        initial=None
        for method in P['methods']:
            previous=np.array([],dtype=int)
            for budget in P['target_budgets']:
                folder=root/f'seed_{seed}'/method/f'labels_{budget}'
                ids=np.asarray(json.loads((folder/'acquired_ids.json').read_text()))
                acquired_labels=np.asarray(json.loads((folder/'acquired_labels.json').read_text()))
                assert len(ids)==budget and len(np.unique(ids))==budget and ((ids>=0)&(ids<pool_size)).all()
                assert len(acquired_labels)==budget
                np.testing.assert_array_equal(acquired_labels,oracle[ids])
                if budget==P['target_budgets'][0]:
                    if initial is None:initial=ids.copy()
                    assert np.array_equal(initial,ids)
                else:
                    assert np.array_equal(ids[:len(previous)],previous)
                previous=ids
                blob=np.load(folder/'development.npz',allow_pickle=False)
                checkpoint=np.load(folder/'model.npz',allow_pickle=False)
                recomputed_probabilities=softmax(multiply(development,checkpoint['coef'].T)+checkpoint['intercept'],axis=1)
                np.testing.assert_allclose(recomputed_probabilities,blob['probabilities'],atol=1e-8)
                result=json.loads((folder/'result.json').read_text())
                assert (np.bincount(blob['targets'],minlength=7)>=P['minimum_development_per_class']).all()
                recomputed=metrics(blob['targets'],blob['probabilities'],blob['sample_weights'])
                for key in ['accuracy','macro_f1','balanced_accuracy','fixed_risk','symmetric_error']:
                    assert np.isclose(recomputed[key],result[key])
                assert result['total_target_training_and_controller_labels']==budget+P['target_controller_labels']
                assert result['development_annotation_overhead']==P['target_development_labels']
                rows+=1
    save(root/'audit.json',{'complete':True,'results_checked':rows,'common_initial_ids_verified':True,
        'metrics_recomputed':True,'probabilities_recomputed_from_models':True,
        'oracle_labels_checked':True,'annotation_overhead_counted':True,'test_evaluated':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(root/'metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,['macro_f1','fixed_risk','balanced_accuracy']):
        for method in P['methods']:
            c=df[df.method.eq(method)].groupby('labels')[metric].agg(['mean','std'])
            x=c.index.to_numpy();mean=c['mean'].to_numpy();std=c['std'].to_numpy()
            ax.plot(x,mean,label=method);ax.fill_between(x,mean-std,mean+std,alpha=.08)
        ax.set_title(metric);ax.set_xlabel('Acquired target labels (+700 additional)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,bbox_to_anchor=(1.03,1),loc='upper left')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);fig.savefig(root/'learning_curves.pdf')
    print('Audited',rows,'results; reserved test not evaluated.',flush=True)


if __name__=='__main__':main()
