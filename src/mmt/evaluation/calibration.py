"""E05 - one scalar temperature, fitted on the CALIBRATION partition only with CPU float64 LBFGS."""

import json
import platform
from pathlib import Path

import numpy as np
import torch

from .io_utils import dump_json
from .metrics import log_softmax

LOG_T_BOUNDS = (-5.0, 5.0)  # T in [0.0067, 148]; hitting a bound is reported, not hidden


class CalibrationError(RuntimeError):
    pass


def _nll_and_grad(log_T, S, y):
    T = np.exp(log_T[0])
    z = S / T
    lp = log_softmax(z)
    p = np.exp(lp)
    n = len(y)
    nll = -lp[np.arange(n), y].mean()
    grad = np.mean(z[np.arange(n), y] - (p * z).sum(1))  # d NLL / d log_T  (dz/dlogT = -z)
    return nll, np.array([grad])


def fit_temperature(scores, y, C, identity, score_kind):
    """scores: [N,C] frozen scores of computable, label-eligible calibration decisions."""
    S = np.asarray(scores, np.float64)
    y = np.asarray(y, int)
    base = dict(
        schema_version=1,
        score_kind=score_kind,
        identity=identity,
        optimizer="torch LBFGS",
        bounds=list(LOG_T_BOUNDS),
        versions=dict(
            numpy=np.__version__, torch=torch.__version__, python=platform.python_version()
        ),
    )
    fallback = dict(base, T=1.0, status="fit_failed")
    if S.ndim != 2 or len(S) == 0 or S.shape[1] != C:
        return dict(fallback, reason="empty or wrongly shaped calibration input")
    if not np.isfinite(S).all():
        return dict(fallback, reason="non-finite scores")
    if ((y < 0) | (y >= C)).any():
        return dict(fallback, reason="labels outside the class map")
    support = {int(c): int((y == c).sum()) for c in range(C)}
    nll0, g0 = _nll_and_grad(np.array([0.0]), S, y)
    scores_tensor = torch.as_tensor(S, dtype=torch.float64, device="cpu")
    targets_tensor = torch.as_tensor(y, dtype=torch.long, device="cpu")
    raw_log_T = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))
    optimizer = torch.optim.LBFGS(
        [raw_log_T],
        lr=1.0,
        max_iter=50,
        history_size=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        log_temperature = LOG_T_BOUNDS[1] * torch.tanh(raw_log_T)
        loss = torch.nn.functional.cross_entropy(
            scores_tensor / log_temperature.exp(), targets_tensor
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite temperature-fitting loss")
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        log_temperature = float((LOG_T_BOUNDS[1] * torch.tanh(raw_log_T.detach())).item())
        T = float(np.exp(log_temperature))
        nll1, g1 = _nll_and_grad(np.array([log_temperature]), S, y)
        optimizer_state = optimizer.state.get(raw_log_T, {})
        iterations = int(optimizer_state.get("n_iter", 0))
        closure_evaluations = int(optimizer_state.get("func_evals", 0))
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return dict(fallback, reason=f"optimizer failed: {type(error).__name__}: {error}")
    converged = abs(float(g1[0])) <= 1e-6
    ok = np.isfinite(T) and T > 0 and np.isfinite(nll1) and nll1 <= nll0 + 1e-8 and converged
    diag = dict(
        initial_nll=float(nll0),
        final_nll=float(nll1),
        initial_grad=float(g0[0]),
        final_grad=float(g1[0]),
        iterations=iterations,
        closure_evaluations=closure_evaluations,
        termination="torch LBFGS step completed",
        converged=converged,
        at_bound=bool(abs(log_temperature) >= LOG_T_BOUNDS[1] - 1e-3),
        class_support=support,
        missing_classes=[c for c, n in support.items() if n == 0],
        n=int(len(y)),
    )
    acc = float((S.argmax(1) == y).mean())
    diag["calibration_accuracy"] = acc
    diag["warning"] = (
        "calibration data are perfectly classified: T is poorly determined, do not trust it"
        if acc == 1.0
        else None
    )
    if not ok:
        return dict(fallback, reason="postcondition failed", diagnostics=diag)
    return dict(base, T=T, status="ok", diagnostics=diag)


def save_calibration(path, art):
    dump_json(path, art)


def load_calibration(path, expected_identity):
    """Reject an artifact if ANY identity field differs from the model being evaluated."""
    art = json.loads(Path(path).read_text())
    for k, v in expected_identity.items():
        if art["identity"].get(k) != v:
            raise CalibrationError(f"calibration identity mismatch on {k!r}")
    return art
