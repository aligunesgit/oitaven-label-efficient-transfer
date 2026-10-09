"""Build spatially separated oracle-simulation samples without test-driven choices."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.optimize import milp, Bounds, LinearConstraint
import shutil
from common import BASE, P, save, split_centers, extract

ROOT = BASE.parent
OUT = BASE / 'prepared'


def write(name, image, gt, coords, blocks, cap, rng, include_labels=True, minimum_per_class=0):
    all_y=gt[coords[:,0],coords[:,1]].astype(int)-1
    population=np.bincount(all_y,minlength=7)
    if cap and len(coords) > cap:
        if minimum_per_class:
            mandatory=[]
            for cls in range(7):
                candidates=np.flatnonzero(all_y==cls)
                assert len(candidates)>=minimum_per_class,f'{name} lacks class {cls} support.'
                mandatory.extend(rng.choice(candidates,minimum_per_class,replace=False).tolist())
            rest=np.setdiff1d(np.arange(len(coords)),mandatory)
            ids=np.r_[mandatory,rng.choice(rest,cap-len(mandatory),replace=False)]
        else:
            ids = rng.choice(len(coords), cap, replace=False)
        coords, blocks = coords[ids], blocks[ids]
    assert len(coords)
    features = extract(image, coords)
    y = gt[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
    assert ((y >= 0) & (y < 7)).all()
    selected_counts=np.bincount(y,minlength=7)
    if minimum_per_class:assert (selected_counts>=minimum_per_class).all()
    weights=population[y]/selected_counts[y]
    parent_side=P['reserved_test_block_pixels']
    parent_ids=coords[:,0]//parent_side*((image.shape[1]+parent_side-1)//parent_side)+coords[:,1]//parent_side
    np.savez_compressed(OUT / f'{name}.npz', X=features, coords=coords, blocks=blocks,
                        parent_blocks=parent_ids,inclusion_weights=weights)
    np.save(OUT / f'{name}_labels.npy', y)
    return {'samples': len(coords), 'blocks': sorted(set(blocks.tolist())), 'parent_blocks':sorted(set(parent_ids.tolist())),
            **({'class_counts': np.bincount(y, minlength=7).tolist()} if include_labels else {})}


def assign_non_test_blocks(blocks,y,seed):
    unique=np.unique(blocks)
    counts=np.stack([np.bincount(y[blocks==b],minlength=7) for b in unique])
    n=len(unique)
    matrix=[];lower=[];upper=[]
    for i in range(n):
        row=np.zeros(n*3);row[i*3:i*3+3]=1
        matrix.append(row);lower.append(1);upper.append(1)
    for group,minimum in enumerate([10,P['minimum_controller_per_class'],P['minimum_development_per_class']]):
        for cls in range(7):
            row=np.zeros(n*3);row[group::3]=counts[:,cls]
            matrix.append(row);lower.append(minimum);upper.append(np.inf)
    for group,fraction in enumerate([.70,.18,.12]):
        row=np.zeros(n*3);row[group::3]=1
        matrix.append(row);lower.append(max(1,int(n*fraction)-4));upper.append(int(n*fraction)+5)
    rng=np.random.default_rng(seed)
    solution=milp(c=rng.uniform(0,1,n*3),integrality=np.ones(n*3),bounds=Bounds(0,1),
        constraints=LinearConstraint(np.stack(matrix),np.asarray(lower),np.asarray(upper)),options={'time_limit':30})
    assert solution.x is not None,f'No class-supported spatial partition: {solution.message}'
    assignment=solution.x.reshape(n,3).argmax(1)
    lookup={int(b):int(g) for b,g in zip(unique,assignment)}
    groups=np.array([lookup[int(b)] for b in blocks])
    for group,minimum in enumerate([10,P['minimum_controller_per_class'],P['minimum_development_per_class']]):
        assert (np.bincount(y[groups==group],minlength=7)>=minimum).all()
    return groups,lookup


def main():
    OUT.mkdir(exist_ok=True)
    signature = hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        assert m['complete'] and m['protocol_sha256'] == signature
        print('Completed preparation retained.', flush=True)
        return
    rng = np.random.default_rng(P['split_seed'])
    source_files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / 'DomainAdaptationDataset/Datasets').glob('*.mat') if p.stem != 'Oiteven'}
    expected = json.loads((ROOT / 'DomainAdaptationDataset_download_check.json').read_text())
    for item in expected['files']:
        name = 'DomainAdaptationDataset/' + item['file']
        if name in source_files:
            assert source_files[name] == item['sha256'], 'Raw dataset changed.'
    summaries = {}
    for domain in ['Source', 'Target']:
        image = loadmat(ROOT / f'DomainAdaptationDataset/Datasets/{domain}.mat')[domain]
        gt = loadmat(ROOT / f'DomainAdaptationDataset/Datasets/{domain}_gt.mat')[domain + '_gt']
        assert image.shape[:2] == gt.shape and image.shape[2] == 5
        assert set(np.unique(gt)) <= set(range(8))
        fractions = [.8, .2] if domain == 'Source' else [1.]
        stride = P['source_stride'] if domain == 'Source' else P['target_stride']
        coords, blocks, groups, assignments = split_centers(gt.shape, stride, fractions, P['split_seed'])
        values = image[coords[:, 0], coords[:, 1]]
        valid = (gt[coords[:, 0], coords[:, 1]] > 0) & np.isfinite(values).all(1) & (values != 0).any(1)
        coords, blocks, groups = coords[valid], blocks[valid], groups[valid]
        if domain == 'Source':
            save(OUT / f'{domain}_block_assignments.json', assignments)
            c, b = coords[groups == 0], blocks[groups == 0]
            y = gt[c[:, 0], c[:, 1]].astype(int) - 1
            chosen = []
            for cls in range(7):
                candidates = np.flatnonzero(y == cls)
                assert len(candidates), f'Source spatial training subset lacks class {cls}; stop rather than alter test split.'
                chosen.extend(rng.choice(candidates, min(len(candidates), P['source_samples_per_class_cap']), replace=False))
            summaries['source_train'] = write('source_train', image, gt, c[chosen], b[chosen], None, rng)
            summaries['source_control'] = write('source_control', image, gt, coords[groups == 1], blocks[groups == 1], P['source_control_cap'], rng)
        else:
            old=BASE/'diagnostics/rejected_geometry_split/prepared'
            frozen_assignments=json.loads((old/'Target_block_assignments.json').read_text())
            side=P['reserved_test_block_pixels']
            parent=coords[:,0]//side*((gt.shape[1]+side-1)//side)+coords[:,1]//side
            keep=np.array([frozen_assignments.get(str(int(b)))!=3 for b in parent])
            coords,blocks=coords[keep],blocks[keep]
            y=gt[coords[:,0],coords[:,1]].astype(int)-1
            groups,assignments=assign_non_test_blocks(blocks,y,P['split_seed'])
            save(OUT/'Target_block_assignments.json',assignments)
            for group, name, cap,minimum in [(0, 'target_pool', None,0),
                                      (1, 'target_controller', P['target_controller_labels'],P['minimum_controller_per_class']),
                                      (2, 'target_development', P['target_development_labels'],P['minimum_development_per_class'])]:
                summaries[name] = write(name, image, gt, coords[groups == group], blocks[groups == group], cap, rng,
                                       include_labels=False,minimum_per_class=minimum)
                if cap:
                    assert summaries[name]['samples'] == cap, f'{name} has too few samples; stop before experiments.'
            for name in ['target_test.npz','target_test_labels.npy']:shutil.copyfile(old/name,OUT/name)
            summaries['target_test']=json.loads((old/'manifest.json').read_text())['sets']['target_test']
            test_parents={int(b) for b,g in frozen_assignments.items() if g==3}
            for name in ['target_pool','target_controller','target_development']:
                assert not set(summaries[name]['parent_blocks'])&test_parents
        print(domain, 'spatial samples prepared.', flush=True)
        del image, gt
    target_names = ['target_pool', 'target_controller', 'target_development']
    for i, name in enumerate(target_names):
        for other in target_names[i+1:]:
            assert not set(summaries[name]['blocks']) & set(summaries[other]['blocks'])
    assert not set(summaries['source_train']['blocks']) & set(summaries['source_control']['blocks'])
    save(manifest_path, {'complete': True, 'protocol_sha256': signature, 'raw_sha256': source_files,
        'sets': summaries, 'patch_radius': P['patch_radius'], 'buffer_pixels': P['buffer_pixels'],
        'zero_label_is_unannotated': True, 'split_uses_only_geometry': False,
        'non_test_class_support_used_for_block_assignment':True,
        'reserved_test_sample_files_unchanged':True,
        'test_labels_not_used_for_training_or_split_tuning': True})


if __name__ == '__main__':
    main()
