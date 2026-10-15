'''E06 - confidence abstention chosen on VALIDATION only. Coverage is always relative to ELIGIBLE decisions.'''
import numpy as np
from .io_utils import dump_json

def risk_coverage(conf, correct, n_eligible):
    '''conf/correct: arrays over COMPUTABLE eligible decisions. One row per candidate threshold.
    Thresholds come from the observed confidences, so tied confidences always stay together.'''
    conf = np.asarray(conf, float); correct = np.asarray(correct, bool); rows = []
    for tau in sorted({0.0} | set(conf.tolist())):
        acc = conf >= tau; k = int(acc.sum())
        rows.append(dict(tau=float(tau), eligible=int(n_eligible), computable=int(len(conf)), accepted=k,
                         coverage=(k / n_eligible if n_eligible else None),
                         risk=(float(1 - correct[acc].mean()) if k else None)))
    return rows

def select_policy(conf, correct, n_eligible, risk_limit=None, min_accepted_count=None, min_accepted_coverage=None,
                  identity=None):
    curve = risk_coverage(conf, correct, n_eligible)
    pol = dict(schema_version=1, operator='>=', constraints=dict(risk_limit=risk_limit,
               minimum_accepted_count=min_accepted_count, minimum_accepted_coverage=min_accepted_coverage),
               identity=identity or {}, tau=None, selected=None, status=None)
    if risk_limit is None:                                   # no requirement defined: do NOT invent one
        pol['status'] = 'disabled_no_constraints'
        return pol, curve
    feas = [r for r in curve if r['accepted'] > 0 and r['risk'] <= risk_limit
            and r['accepted'] >= (min_accepted_count or 0) and (r['coverage'] or 0) >= (min_accepted_coverage or 0)]
    if not feas:
        pol['status'] = 'no_feasible_policy'
        return pol, curve
    best = sorted(feas, key=lambda r: (-r['coverage'], r['risk'], r['tau']))[0]   # max coverage, then risk, then tau
    pol.update(tau=best['tau'], selected=best, status='selected')
    return pol, curve

def apply_policy(conf, computable, tau):
    '''Identical semantics for curves, selection and deployment: accepted iff computable and conf >= tau.'''
    conf = np.asarray(conf, float); computable = np.asarray(computable, bool)
    if tau is None:
        return computable.copy()
    return computable & (conf >= tau)

def save_policy(path, pol):
    dump_json(path, pol)
