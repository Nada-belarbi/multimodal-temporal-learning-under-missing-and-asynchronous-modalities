'''E07 - model comparison with the INDEPENDENT unit (participant/video) as the resampling unit.
Per-group confusion matrices make each bootstrap replicate a cheap sum, with identical draws for paired models.'''
import numpy as np
import pandas as pd

def macro_f1_from_cm(cm):
    if cm.sum() == 0: return None                                  # undefined -> null replicate
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)))   # zero_division=0, fixed classes

def group_confusions(j, C, ids=None):
    '''j needs group_id, target_class, pred, decision_id. ids restricts to a decision subset (e.g. intersection).'''
    S = j[j.score_eligible & j.computable & j.pred.notna()] if 'pred' in j else j.iloc[0:0]
    if ids is not None: S = S[S.decision_id.isin(ids)]
    out = {}
    for g, d in S.groupby('group_id'):
        cm = np.zeros((C, C), int)
        np.add.at(cm, (d.target_class.to_numpy(int), d.pred.to_numpy(int)), 1); out[g] = cm
    return out

def intersection_ids(jA, jB):
    a = set(jA[jA.score_eligible & jA.computable].decision_id); b = set(jB[jB.score_eligible & jB.computable].decision_id)
    return a & b

def paired_bootstrap(cmsA, cmsB, C, metric=macro_f1_from_cm, B=1000, seed=2026):
    groups = sorted(set(cmsA) | set(cmsB)); G = len(groups); zero = np.zeros((C, C), int)
    if G < 2:
        return dict(status='suppressed_fewer_than_two_groups', n_groups=G)
    A = np.stack([cmsA.get(g, zero) for g in groups]); Bm = np.stack([cmsB.get(g, zero) for g in groups])
    rng = np.random.Generator(np.random.PCG64(seed)); draws = rng.integers(0, G, size=(B, G))   # same draw for both models
    da, db = [], []
    for k in range(B):
        a, b = metric(A[draws[k]].sum(0)), metric(Bm[draws[k]].sum(0))
        da.append(a); db.append(b)
    diff = [None if (a is None or b is None) else a - b for a, b in zip(da, db)]
    ok = np.array([d for d in diff if d is not None], float)
    pa, pb = metric(A.sum(0)), metric(Bm.sum(0))
    res = dict(n_groups=G, replicates=B, valid_replicates=int(len(ok)), seed=seed, observed_A=pa, observed_B=pb,
               observed_diff=None if pa is None or pb is None else pa - pb, resampled_group_indices=draws.tolist(),
               note='descriptive and unstable: few independent groups' if G < 10 else 'ok')
    if len(ok) == 0:
        return dict(res, status='all_replicates_undefined', ci95=None)
    return dict(res, status='ok', ci95=[float(x) for x in np.percentile(ok, [2.5, 97.5])])

def seed_summary(values):
    v = [x for x in values if x is not None]
    return dict(n=len(v), values=values, mean=float(np.mean(v)) if v else None,
                std=float(np.std(v, ddof=1)) if len(v) > 1 else None)       # sample std, ddof=1; never the best seed

def matched_coverage(curveA, curveB, target):
    '''Only report equal coverage where attainable; otherwise show the neighbouring attainable points.'''
    def near(curve):
        pts = [r for r in curve if r['coverage'] is not None and r['accepted'] > 0]
        exact = [r for r in pts if abs(r['coverage'] - target) < 1e-12]
        if exact: return dict(exact=exact[0])
        lo = [r for r in pts if r['coverage'] < target]; hi = [r for r in pts if r['coverage'] > target]
        return dict(below=max(lo, key=lambda r: r['coverage']) if lo else None,
                    above=min(hi, key=lambda r: r['coverage']) if hi else None)
    a, b = near(curveA), near(curveB)
    return dict(target=target, A=a, B=b, matched=('exact' in a and 'exact' in b))
