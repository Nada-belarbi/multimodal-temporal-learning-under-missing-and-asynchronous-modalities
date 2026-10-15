'''E04 - evaluation runner. Verifies identities, applies a SAVED schedule, asks the model for one status
per scheduled decision, joins labels INSIDE the evaluator, writes results atomically.'''
import numpy as np, pandas as pd, platform, sys
from pathlib import Path
from .io_utils import AtomicDir, dump_json, canonical_hash
from .metrics import join_registry, compute_metrics, log_probs, per_group
from .perturbations import apply_schedule

class IdentityError(RuntimeError): pass
class ModelFailure(RuntimeError): pass

def verify_identities(expected, actual):
    '''Every hash named in `expected` must match. A mismatch stops the run before any prediction.'''
    bad = {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
    if bad:
        raise IdentityError(f'identity mismatch: {bad}')

def status_for(computable, conf, tau):
    if not computable: return 'unavailable_no_data'
    if tau is not None and conf < tau: return 'abstained_low_confidence'
    return 'accepted'

def run_predictions(registry, sources, schedule, predict_fn, T=1.0, tau=None):
    '''One returned row per scheduled decision (eligible or not). Labels are never passed to the model.'''
    work = apply_schedule(sources, schedule) if schedule else sources
    by_seq = {}
    for (seq, sid), s in work.items(): by_seq.setdefault(seq, {})[sid] = s
    out = []
    for r in registry.itertuples(index=False):
        row = dict(decision_id=r.decision_id, sequence_id=r.sequence_id, decision_time=r.decision_time,
                   support_start=r.support_start, support_end=r.support_end)      # no label fields
        try:
            p = predict_fn(row, by_seq[r.sequence_id])
        except Exception as e:                                                      # model error = FAILED run
            raise ModelFailure(f'decision {r.decision_id}: {type(e).__name__}: {e}') from e
        conf, scores = None, p.get('scores')
        if p['computable'] and scores is not None:
            lp = log_probs(np.asarray(scores)[None, :], T)[0]; conf = float(np.exp(lp).max())
        pc = p.get('predicted_class')
        out.append(dict(decision_id=r.decision_id, computable=bool(p['computable']), scores=scores, predicted_class=pc,
                        confidence=conf, source_presence=p.get('presence'),
                        status=status_for(p['computable'], conf if conf is not None else 1.0, tau),
                        accepted=status_for(p['computable'], conf if conf is not None else 1.0, tau) == 'accepted'))
    return pd.DataFrame(out)

def evaluate(registry, sources, schedule, predict_fn, C, out_dir, run_id, scenario_id, score_kind,
             calibration=None, policy=None, expected_ids=None, actual_ids=None, partition=None):
    if expected_ids: verify_identities(expected_ids, actual_ids or {})
    reg = registry if partition is None else registry[registry.split == partition].reset_index(drop=True)
    T = 1.0 if not calibration or calibration.get('status') != 'ok' else calibration['T']
    tau = None if not policy else policy.get('tau')
    preds = run_predictions(reg, sources, schedule, predict_fn, T, tau)
    j = join_registry(reg, preds, C)
    m = compute_metrics(j, C, score_kind, T)
    if len(j[j.score_eligible & j.computable]) and 'scores' in j:
        S = j[j.score_eligible & j.computable]
        if S.scores.notna().all():
            pred = np.vstack(S.scores.to_numpy()); j.loc[S.index, 'pred'] = log_probs(pred, T).argmax(1)
    prov = dict(run_id=run_id, scenario_id=scenario_id, partition=partition, T=T, tau=tau, score_kind=score_kind,
                identities=actual_ids or {}, schedule_hash=(schedule or {}).get('schedule_hash'),
                schedule_stats=(schedule or {}).get('stats'), calibration_status=(calibration or {}).get('status', 'none'),
                python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform())
    with AtomicDir(out_dir) as tmp:
        _write(tmp, j, m, prov)
    return m, j

def _write(tmp, j, m, prov):
    tmp = Path(tmp); cols = [c for c in j.columns if c != 'scores']
    flat = j[cols].copy(); flat['scores'] = j['scores'].map(lambda s: None if s is None else ';'.join(f'{x:.9g}' for x in s))
    try:
        flat.to_parquet(tmp / 'predictions.parquet'); fmt = 'parquet'
    except Exception:                                   # pyarrow missing: documented CSV fallback
        flat.to_csv(tmp / 'predictions.csv', index=False); fmt = 'csv'
    ex = m['extra']
    if 'class_metrics' in ex: ex['class_metrics'].to_csv(tmp / 'class_metrics.csv', index=False)
    if 'confusion' in ex: pd.DataFrame(ex['confusion']).to_csv(tmp / 'confusion.csv', index=False)
    if 'reliability' in ex: pd.DataFrame(ex['reliability']).to_csv(tmp / 'reliability.csv', index=False)
    pg = per_group(j)
    if len(pg): pg.to_csv(tmp / 'per_group.csv', index=False)
    clean = {k: v for k, v in m.items() if k != 'extra'}
    dump_json(tmp / 'metrics.json', dict(schema_version=1, prediction_format=fmt, **clean))
    dump_json(tmp / 'provenance.json', prov)
    dump_json(tmp / 'status.json', dict(status='completed', rows=len(j)))
