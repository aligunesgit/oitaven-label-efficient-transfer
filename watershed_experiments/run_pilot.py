"""Matched fixed/adaptive-cost pilot. Reserved test files are never opened."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import BASE, P, save, fit, metrics, coral, adaptive_weights, choose, multiply

OUT = BASE / 'pilot'


class Oracle:
    def __init__(self, path):
        self._labels = np.load(path, mmap_mode='r', allow_pickle=False)
        self.acquired = set()

    def query(self, ids):
        ids = np.asarray(ids, dtype=int)
        assert len(np.unique(ids)) == len(ids)
        assert not self.acquired.intersection(ids.tolist()), 'Duplicate annotation charged.'
        self.acquired.update(ids.tolist())
        return np.array(self._labels[ids], copy=True)


def load(name):
    assert name != 'target_test', 'Reserved test must not be read by pilot.'
    return np.load(BASE / 'prepared' / f'{name}.npz', allow_pickle=False)['X'].astype(float)


def labels(name):
    assert name in ['source_train', 'source_control', 'target_controller', 'target_development']
    return np.load(BASE / 'prepared' / f'{name}_labels.npy', allow_pickle=False)


def fixed_weights(method):
    coefficient = int(method[-1]) if method.startswith('fixed_risk') else 3
    weights = np.ones(7)
    weights[P['fixed_coefficient_classes']] = coefficient
    return weights


def report():
    rows = []
    for path in OUT.glob('seed_*/*/labels_*/result.json'):
        result = json.loads(path.read_text())
        rows.append({k: v for k, v in result.items() if k not in ['recalls', 'multipliers', 'controller_support', 'controller_fnr']})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(['seed', 'method', 'labels'])
    df.to_csv(OUT / 'metrics.csv', index=False)
    summary = df.groupby(['method', 'labels'])[['accuracy', 'macro_f1', 'balanced_accuracy', 'fixed_risk']].agg(['mean', 'std', 'count'])
    summary.to_csv(OUT / 'summary.csv')
    lines = ['# Cost-sensitive active regional adaptation pilot', '',
             'Revised class-supported spatial protocol: original test blocks frozen,256-pixel non-test blocks with32-pixel buffers. Non-test benchmark labels are used to construct class-supported subsets; acquisition never sees unacquired labels. Controller/development samples are enriched to at least20 per class; development metrics are inverse-inclusion weighted. The rejected geometric pilot is diagnostic only.', '',
             'Development evaluation only; reserved test never opened. Five-band local spectral/spatial features, source-only standardization, CORAL fit on source training and unlabeled adaptation pool, multinomial logistic classifier.', '',
             'Source class-balanced subsampling assumes labeled source availability. Target controller set costs700 labels, additional to acquisition budgets. Development evaluation uses2000 further labels. Zero labels are unannotated. Sampling is limited to a geometric grid within available reference support.', '',
             'Hypothetical fixed miss costs [3,3,3,1,1,3,1] are used for every reported method and decision. They are not measured ecological losses. Fixed coefficients alter training/acquisition emphasis on water and artificial surfaces; adaptive coefficients can vary per class. Class balance and feature transforms are shared. Probability estimates are not claimed to be calibrated.', '',
             'Fixed/Adaptive Entropy and Risk variants separate acquisition from fitting effects. Adaptive coefficients use only the disjoint700-label controller set, with minimum10 examples per class. Initial query IDs are matched across methods; initial fitted models differ where cost weights differ.', '',
             '| Method | Acquired labels | Seeds | Macro-F1 | Balanced accuracy | Fixed risk | Accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for (method, budget), row in summary.iterrows():
        lines.append(f'| {method} | {budget} | {int(row[("accuracy","count")])} | ' +
                     ' | '.join(f'{row[(metric,"mean")]:.4f}' for metric in ['macro_f1','balanced_accuracy','fixed_risk','accuracy']) + ' |')
    lines += ['', 'Single-scene pilot; seeds share one spatial split. Buffer/scale sensitivity and independent-scene confirmation remain necessary. No LLM is used and no publication-ready superiority is claimed.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    files = [BASE / name for name in ['protocol.json','common.py','prepare.py','run_pilot.py']]
    hashes = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    freeze = OUT / 'source_hashes.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == hashes, 'Code/protocol changed; use a new output folder.'
    else:
        save(freeze, hashes)
    manifest = json.loads((BASE / 'prepared/manifest.json').read_text())
    assert manifest['complete']
    scaler = StandardScaler().fit(load('source_train'))
    source = scaler.transform(load('source_train'))
    control_source = scaler.transform(load('source_control'))
    pool = scaler.transform(load('target_pool'))
    controller = scaler.transform(load('target_controller'))
    development = scaler.transform(load('target_development'))
    ys, ysc, yc, yd = [labels(name) for name in ['source_train','source_control','target_controller','target_development']]
    assert (np.bincount(yd,minlength=7)>=P['minimum_development_per_class']).all()
    assert (np.bincount(yc,minlength=7)>=P['minimum_controller_per_class']).all()
    development_weights=np.load(BASE/'prepared/target_development.npz',allow_pickle=False)['inclusion_weights']
    adapted, mean_s, mean_t, mapping = coral(source, pool)
    np.savez_compressed(OUT / 'transforms.npz', scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        source_mean=mean_s, target_mean=mean_t, coral_mapping=mapping)
    base_weights = np.asarray(P['evaluation_miss_cost'])
    baselines = {}
    for name, Xs, Xsc in [('source_only',source,control_source),
                          ('coral_unlabeled',adapted,multiply(control_source-mean_s, mapping)+mean_t)]:
        model = fit(Xs, ys, base_weights)
        probabilities = model.predict_proba(development)
        np.savez_compressed(OUT / f'{name}_development.npz', probabilities=probabilities, targets=yd,sample_weights=development_weights)
        baselines[name] = {'development': metrics(yd, probabilities,development_weights),
                           'source_control': metrics(ysc, model.predict_proba(Xsc)), 'target_acquired_labels': 0}
    save(OUT / 'baselines.json', baselines)
    try:
        for seed in P['seeds']:
            initial = np.random.default_rng(seed).choice(len(pool), P['target_budgets'][0], replace=False)
            for method in P['methods']:
                save(OUT / 'status.json', {'state':'running','seed':seed,'method':method})
                oracle = Oracle(BASE / 'prepared/target_pool_labels.npy')
                acquired = initial.copy()
                ya = oracle.query(acquired)
                multiplier = np.ones(7) if method.startswith('adaptive') else fixed_weights(method)
                for step, budget in enumerate(P['target_budgets']):
                    folder = OUT / f'seed_{seed}' / method / f'labels_{budget}'
                    folder.mkdir(parents=True, exist_ok=True)
                    start = time.time()
                    X_train, y_train = np.r_[adapted,pool[acquired]], np.r_[ys,ya]
                    if method.startswith('budgetmatched'):
                        X_train, y_train = np.r_[X_train,controller], np.r_[y_train,yc]
                    model = fit(X_train, y_train, base_weights * multiplier)
                    probabilities = model.predict_proba(development)
                    np.savez_compressed(folder/'model.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
                    result = {'seed':seed,'method':method,'labels':len(acquired),
                              'total_target_training_and_controller_labels':len(acquired)+len(yc),
                              'development_annotation_overhead':len(yd),
                              'multipliers':multiplier.tolist(), **metrics(yd,probabilities,development_weights)}
                    np.savez_compressed(folder / 'development.npz', probabilities=probabilities, targets=yd,sample_weights=development_weights)
                    save(folder / 'acquired_ids.json', acquired.tolist())
                    save(folder / 'acquired_labels.json', ya.tolist())
                    controller_probs = model.predict_proba(controller)
                    next_weights, support, fnr = adaptive_weights(yc, controller_probs)
                    result.update(controller_support=support.tolist(), controller_fnr=fnr.tolist(),
                                  elapsed_seconds=time.time()-start)
                    save(folder / 'result.json', result)
                    print(f'seed{seed} {method} labels{budget}: macroF1={result["macro_f1"]:.4f}, fixed risk={result["fixed_risk"]:.4f}',flush=True)
                    report()
                    if step == len(P['target_budgets'])-1:
                        continue
                    if method.startswith('adaptive'):
                        multiplier = next_weights
                    amount = P['target_budgets'][step+1]-len(acquired)
                    available = np.setdiff1d(np.arange(len(pool)),acquired)
                    rng = np.random.default_rng(seed+10000+step)
                    chosen = choose(method,model.predict_proba(pool[available]),available,amount,multiplier,rng)
                    save(folder / 'next_query.json', {'ids':chosen.tolist(),'hidden_labels_used':False,
                         'controller_multipliers':multiplier.tolist(),'available_count':len(available)})
                    ya = np.r_[ya,oracle.query(chosen)]
                    acquired = np.r_[acquired,chosen]
        report()
        save(OUT / 'status.json', {'state':'complete','test_evaluated':False})
    except Exception as error:
        save(OUT / 'status.json',{'state':'failed','error':str(error)})
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
