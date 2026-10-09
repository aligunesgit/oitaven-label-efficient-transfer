import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

BASE = Path(__file__).resolve().parent
P = json.loads((BASE / 'protocol.json').read_text())


def multiply(a, b):
    # Explicit finite matrix contraction avoids spurious FPE flags from local Accelerate matmul.
    result = np.einsum('ij,jk->ik', a, b, optimize=False)
    assert np.isfinite(result).all()
    return result


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def split_centers(shape, stride, fractions, seed):
    block, buffer = P['block_pixels'], P['buffer_pixels']
    radius = P['patch_radius']
    rr, cc = np.meshgrid(np.arange(buffer, shape[0]-buffer, stride),
                         np.arange(buffer, shape[1]-buffer, stride), indexing='ij')
    coords = np.c_[rr.ravel(), cc.ravel()]
    # Local centers stay away from every block edge; patches cannot cross subsets.
    keep = ((coords % block >= buffer) & (coords % block < block-buffer)).all(1)
    coords = coords[keep]
    nbcols = (shape[1] + block-1)//block
    block_ids = coords[:, 0]//block * nbcols + coords[:, 1]//block
    unique = np.unique(block_ids)
    permutation = np.random.default_rng(seed).permutation(unique)
    boundaries = np.r_[0, np.cumsum(np.asarray(fractions) * len(unique)).astype(int)]
    boundaries[-1] = len(unique)
    assignments = {}
    for group in range(len(fractions)):
        for b in permutation[boundaries[group]:boundaries[group+1]]:
            assignments[int(b)] = group
    groups = np.array([assignments[int(b)] for b in block_ids])
    assert (coords.min(1) >= radius).all()
    return coords, block_ids, groups, assignments


def extract(image, coords):
    radius = P['patch_radius']
    features = np.empty((len(coords), 15), dtype=np.float32)
    for index, (row, col) in enumerate(coords):
        patch = image[row-radius:row+radius+1, col-radius:col+radius+1].astype(np.float32)
        features[index] = np.r_[image[row, col], patch.mean((0, 1)), patch.std((0, 1))]
    assert np.isfinite(features).all()
    return features


def cost_matrix(multiplier=None):
    penalty = np.asarray(P['evaluation_miss_cost'], dtype=float)
    if multiplier is not None:
        penalty = penalty * np.asarray(multiplier)
    matrix = np.repeat(penalty[:, None], 7, axis=1)
    np.fill_diagonal(matrix, 0)
    return matrix


def decisions(probabilities):
    return multiply(probabilities, cost_matrix()).argmin(1)


def metrics(y, probabilities, sample_weight=None):
    assert probabilities.shape==(len(y),7)
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-6)
    pred = decisions(probabilities)
    recalls = recall_score(y, pred, labels=np.arange(7), average=None, zero_division=0,sample_weight=sample_weight)
    present=np.bincount(y,minlength=7)>0
    return {'accuracy': float(accuracy_score(y, pred,sample_weight=sample_weight)),
            'macro_f1': float(f1_score(y, pred, labels=np.arange(7), average='macro', zero_division=0,sample_weight=sample_weight)),
            'balanced_accuracy': float(recalls[present].mean()),
            'fixed_risk': float(np.average(cost_matrix()[y, pred],weights=sample_weight)),
            'symmetric_error': float(np.average(y != pred,weights=sample_weight)),
            'recalls': recalls.tolist()}


def fit(X, y, weights):
    assert set(np.unique(y)) == set(range(7)), 'Training must represent every source class.'
    counts = np.bincount(y, minlength=7)
    sample_weights = np.asarray(weights)[y] / counts[y]
    sample_weights *= len(y) / sample_weights.sum()
    model = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    model.fit(X, y, sample_weight=sample_weights)
    assert np.array_equal(model.classes_, np.arange(7))
    assert model.n_iter_.max() < 2000, 'Optimizer did not converge.'
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    return model


def adaptive_weights(y, probabilities):
    predicted = decisions(probabilities)
    support = np.bincount(y, minlength=7)
    fn = np.bincount(y[y != predicted], minlength=7)
    rates = (fn + 1) / (support + 2)
    weights = np.select([rates <= .10, rates <= .25, rates <= .50], [1., 3., 5.], default=7.)
    weights[support < 10] = 1.
    return weights, support, rates


def choose(method, probabilities, available, amount, weights, rng):
    assert amount <= len(available)
    if method == 'fixed_random':
        chosen = rng.choice(available, amount, replace=False)
    else:
        if method.endswith('entropy'):
            scores = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(1)
        else:
            scores = multiply(probabilities, cost_matrix(weights)).min(1)
        chosen = available[np.argsort(-scores, kind='stable')[:amount]]
    assert len(np.unique(chosen)) == amount and np.isin(chosen, available).all()
    return chosen


def covariance_power(X, exponent):
    covariance = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-3
    values, vectors = np.linalg.eigh(covariance)
    assert (values > 0).all()
    return multiply(vectors * values**exponent, vectors.T)


def coral(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    mapping = multiply(covariance_power(source, -.5), covariance_power(target, .5))
    transformed = multiply(source-source_mean, mapping) + target_mean
    assert np.isfinite(transformed).all()
    return transformed, source_mean, target_mean, mapping
