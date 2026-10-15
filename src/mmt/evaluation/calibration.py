'''E05 - one scalar temperature, fitted on the CALIBRATION partition only.
Tested path uses SciPy L-BFGS-B on log_T with an analytic gradient. The spec's torch LBFGS version is in
calibration_torch.py (not run here: torch was unavailable).'''
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from .io_utils import dump_json, canonical_hash
from .metrics import log_softmax
import json, platform, scipy

LOG_T_BOUNDS = (-5.0, 5.0)        # T in [0.0067, 148]; hitting a bound is reported, not hidden

class CalibrationError(RuntimeError): pass

def _nll_and_grad(log_T, S, y):
    T = np.exp(log_T[0]); z = S / T; lp = log_softmax(z); p = np.exp(lp); n = len(y)
    nll = -lp[np.arange(n), y].mean()
    grad = np.mean(z[np.arange(n), y] - (p * z).sum(1))        # d NLL / d log_T  (dz/dlogT = -z)
    return nll, np.array([grad])

def fit_temperature(scores, y, C, identity, score_kind):
    '''scores: [N,C] frozen scores of computable, label-eligible calibration decisions.'''
    S = np.asarray(scores, np.float64); y = np.asarray(y, int)
    base = dict(schema_version=1, score_kind=score_kind, identity=identity, optimizer='scipy L-BFGS-B',
                bounds=list(LOG_T_BOUNDS), versions=dict(numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version()))
    fallback = dict(base, T=1.0, status='fit_failed')
    if S.ndim != 2 or len(S) == 0 or S.shape[1] != C:
        return dict(fallback, reason='empty or wrongly shaped calibration input')
    if not np.isfinite(S).all():
        return dict(fallback, reason='non-finite scores')
    if ((y < 0) | (y >= C)).any():
        return dict(fallback, reason='labels outside the class map')
    support = {int(c): int((y == c).sum()) for c in range(C)}
    nll0, g0 = _nll_and_grad(np.array([0.0]), S, y)
    r = minimize(_nll_and_grad, np.array([0.0]), args=(S, y), jac=True, method='L-BFGS-B', bounds=[LOG_T_BOUNDS],
                 options=dict(maxiter=50, ftol=1e-12, gtol=1e-7))
    T = float(np.exp(r.x[0])); nll1, g1 = _nll_and_grad(r.x, S, y)
    ok = np.isfinite(T) and T > 0 and np.isfinite(nll1) and nll1 <= nll0 + 1e-8
    diag = dict(initial_nll=float(nll0), final_nll=float(nll1), initial_grad=float(g0[0]), final_grad=float(g1[0]),
                iterations=int(r.nit), termination=str(r.message), at_bound=bool(abs(r.x[0]) >= LOG_T_BOUNDS[1] - 1e-9),
                class_support=support, missing_classes=[c for c, n in support.items() if n == 0], n=int(len(y)))
    acc = float((S.argmax(1) == y).mean())
    diag['calibration_accuracy'] = acc
    diag['warning'] = ('calibration data are perfectly classified: T is poorly determined, do not trust it'
                       if acc == 1.0 else None)
    if not ok:
        return dict(fallback, reason='postcondition failed', diagnostics=diag)
    return dict(base, T=T, status='ok', diagnostics=diag)

def save_calibration(path, art):
    dump_json(path, art)

def load_calibration(path, expected_identity):
    '''Reject an artifact if ANY identity field differs from the model being evaluated.'''
    art = json.loads(Path(path).read_text())
    for k, v in expected_identity.items():
        if art['identity'].get(k) != v:
            raise CalibrationError(f'calibration identity mismatch on {k!r}')
    return art
