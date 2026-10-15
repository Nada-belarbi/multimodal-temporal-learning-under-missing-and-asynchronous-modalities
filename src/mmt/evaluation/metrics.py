'''E02 - registry join and metrics. Denominators are explicit; empty cases give null, never zero error.'''
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

N_BINS = 15

def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def log_probs(scores, T=1.0):
    '''Same formula for logits and log-probabilities: log_softmax(scores / T). T=1 = uncalibrated.'''
    return log_softmax(np.asarray(scores, np.float64) / T)

# ---------------------------------------------------------------- registry join
PRED_COLS = {'decision_id', 'computable', 'status'}

def join_registry(registry, preds, C):
    '''LEFT JOIN predictions onto the registry (never the reverse). Exactly one row per decision.'''
    if not PRED_COLS <= set(preds.columns):
        raise ValueError(f'predictions need columns {sorted(PRED_COLS)}')
    if preds.decision_id.duplicated().any():
        raise ValueError('duplicate decision ids in predictions')
    unknown = set(preds.decision_id) - set(registry.decision_id)
    if unknown:
        raise ValueError(f'predictions for unknown decisions: {sorted(unknown)[:5]}')
    missing = set(registry.decision_id) - set(preds.decision_id)
    if missing:
        raise ValueError(f'{len(missing)} scheduled decisions have no returned status')
    if 'scores' in preds:
        for s in preds.scores.dropna():
            if len(s) != C:
                raise ValueError(f'score vector length {len(s)} != number of classes {C}')
    j = registry.merge(preds, on='decision_id', how='left', validate='one_to_one')
    j['computable'] = j.computable.astype(bool)
    j['score_eligible'] = j.score_eligible.astype(bool)
    if 'accepted' not in j:
        j['accepted'] = j.computable
    j['accepted'] = j.accepted.astype(bool) & j.computable
    return j

# ---------------------------------------------------------------- calibration pieces
def ece_table(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.searchsorted(edges[1:-1], conf, side='right')      # [j/15,(j+1)/15); last bin includes 1.0
    rows, ece, n = [], 0.0, len(conf)
    for j in range(n_bins):
        m = b == j
        if m.any():
            mc, acc = float(conf[m].mean()), float(correct[m].mean())
            ece += m.sum() / n * abs(mc - acc)
            rows.append(dict(bin=j, lo=edges[j], hi=edges[j + 1], count=int(m.sum()), mean_confidence=mc, accuracy=acc))
        else:
            rows.append(dict(bin=j, lo=edges[j], hi=edges[j + 1], count=0, mean_confidence=None, accuracy=None))
    return float(ece), rows

def probability_metrics(scores, y, T=1.0):
    lp = log_probs(scores, T); p = np.exp(lp); n, C = p.shape
    pred = p.argmax(1); conf = p.max(1); correct = pred == y
    ece, bins = ece_table(conf, correct)
    return dict(nll=float(-lp[np.arange(n), y].mean()),
                brier=float(((p - np.eye(C)[y]) ** 2).sum(1).mean()),   # multiclass; twice the binary convention
                ece=ece, mean_entropy=float(-(p * lp).sum(1).mean()), mean_confidence=float(conf.mean())), bins

# ---------------------------------------------------------------- classification pieces
def classification_metrics(y, pred, C):
    labels = list(range(C))
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    supported = [c for c in labels if sup[c] > 0]
    out = dict(macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
               balanced_accuracy=float(np.mean([r[c] for c in supported])),
               balanced_accuracy_classes=supported, absent_classes=[c for c in labels if sup[c] == 0],
               accuracy=float((y == pred).mean()))
    table = pd.DataFrame(dict(class_index=labels, precision=p, recall=r, f1=f, support=sup))
    return out, table, cm

def _null_quality():
    return dict(macro_f1=None, balanced_accuracy=None, accuracy=None, balanced_accuracy_classes=[], absent_classes=[])

# ---------------------------------------------------------------- main entry
def compute_metrics(j, C, score_kind=None, T=1.0):
    '''j = joined registry+predictions. E eligible, S eligible&computable, A accepted subset of S.'''
    E = j[j.score_eligible]; S = E[E.computable]; A = S[S.accepted]
    counts = dict(scheduled=len(j), eligible=len(E), computable=len(S), accepted=len(A),
                  excluded_by_reason=(j[~j.score_eligible].exclusion_reason.fillna('unknown').value_counts().to_dict()
                                      if 'exclusion_reason' in j else {}),
                  statuses=j.status.value_counts().to_dict())
    cov = dict(availability_coverage=None, accepted_coverage=None, selective_risk=None, status='ok')
    if len(E) == 0:
        cov['status'] = 'no_eligible_labels'
    else:
        cov['availability_coverage'] = len(S) / len(E)
        cov['accepted_coverage'] = len(A) / len(E)
    has_scores = 'scores' in S and len(S) > 0 and S.scores.notna().all()
    res = dict(counts=counts, coverage=cov, quality_S=_null_quality(), quality_A=_null_quality(),
               probability=dict(status='not_available', reason='no computable decisions'), extra={})
    if len(S) == 0:
        return res
    y = S.target_class.to_numpy(int)
    if has_scores:
        M = np.vstack(S.scores.to_numpy())
        lp = log_probs(M, T); pred = lp.argmax(1)               # ties -> lowest class index
        pm, bins = probability_metrics(M, y, T)
        res['probability'] = dict(status='ok', T=T, score_kind=score_kind, **pm)
        res['extra']['reliability'] = bins
    else:
        pred = S.predicted_class.to_numpy(int)
        res['probability'] = dict(status='not_available', reason='label-only reference')
    q, table, cm = classification_metrics(y, pred, C)
    res['quality_S'] = q; res['extra'].update(class_metrics=table, confusion=cm)
    S = S.assign(pred=pred)
    Aidx = S.accepted.to_numpy()
    if Aidx.sum() > 0:
        yA, pA = y[Aidx], pred[Aidx]
        res['quality_A'], _, _ = classification_metrics(yA, pA, C)
        res['coverage']['selective_risk'] = float((yA != pA).mean())   # None when nothing accepted - never 0
    return res

def per_group(j):
    '''Pooled metrics can hide failures on short groups, so export per-group results too.'''
    S = j[j.score_eligible & j.computable]
    if 'pred' not in S:
        return pd.DataFrame()
    return (S.assign(ok=(S.pred == S.target_class))
            .groupby('group_id').agg(decisions=('ok', 'size'), accuracy=('ok', 'mean')).reset_index())
