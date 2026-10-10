"""Step 7: calibration and the accept/escalate/flag decision policy, on val_id only.

docs/PREREGISTRATION.md, "Fitting and selection" (Calibration) and "Decision policy":
- Platt scaling and isotonic regression are compared by expected calibration error (10 equal-width
  bins on P(hallucinated)), cross-fitted within val_id: 5 folds grouped by cluster_id, each method
  fitted on 4 folds and measured on the fifth, averaged over folds and the three training seeds.
  The lower mean ECE wins; a difference under 0.005 goes to Platt. The chosen method is then
  refitted on all of val_id.
- Accept if P < t_low, the largest candidate at which at most 5% of accepted answers are
  hallucinated; escalate if t_low <= P < t_high, the largest candidate at which at most 20% of
  traffic is escalated; flag if P >= t_high. Candidates are the distinct calibrated values on
  val_id. With no feasible t_low the accept band is empty and the policy is reported infeasible.

Calibration never moves the F1 decision threshold, which stays on the uncalibrated score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from threadpoolctl import threadpool_limits

from hdetect.training.models import folds

METHODS = ("platt", "isotonic")
_SKLEARN_METHOD = {"platt": "sigmoid", "isotonic": "isotonic"}


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins on [0, 1]; P = 1 falls in the last bin."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": b, "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed_hallucinated": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def fit_calibrator(estimator, method: str, X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    """scikit-learn's calibrator on the frozen classifier; the classifier itself is never refitted."""
    with threadpool_limits(1):
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=_SKLEARN_METHOD[method]).fit(X, y)


def calibrated(calibrator: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    with threadpool_limits(1):
        return calibrator.predict_proba(X)[:, 1]


def cross_fit(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int,
              n_bins: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-fold ECE for each method, and each method's out-of-fold calibrated probabilities."""
    rows, oof = [], {m: np.full(len(y), np.nan) for m in METHODS}
    for k, (tr, te) in enumerate(folds(y, groups, seed, n_splits)):
        for method in METHODS:
            p = calibrated(fit_calibrator(estimator, method, X[tr], y[tr]), X[te])
            oof[method][te] = p
            rows.append({"seed": seed, "fold": k, "method": method, "n": len(te), "ece": ece(p, y[te], n_bins)})
    return pd.DataFrame(rows), oof


def choose_method(mean_ece: Mapping[str, float], platt_margin: float) -> str:
    """Lower mean ECE wins; isotonic must beat Platt by at least `platt_margin`."""
    return "isotonic" if mean_ece["platt"] - mean_ece["isotonic"] >= platt_margin else "platt"


@dataclass(frozen=True)
class Policy:
    t_low: float
    t_high: float
    feasible: bool
    n: int
    n_distinct: int
    accept_rate: float
    accept_hallucinated_share: float
    escalate_rate: float
    flag_rate: float
    flag_hallucinated_share: float
    plateau_share_t_low: float
    plateau_share_t_high: float

    def band(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.where(p < self.t_low, "accept", np.where(p < self.t_high, "escalate", "flag"))

    def to_dict(self) -> dict:
        return asdict(self)


def policy_thresholds(p: np.ndarray, y: np.ndarray, accept_max_hallucinated: float,
                      escalate_max_share: float) -> Policy:
    """Goals are compared as exact fractions, so 5% of 20 accepted answers allows exactly one."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    n = len(p)
    cands = np.unique(p)
    a_goal, e_goal = Fraction(str(accept_max_hallucinated)), Fraction(str(escalate_max_share))

    t_low, feasible = float(cands[0]), False
    for c in cands[::-1]:
        acc = p < c
        if acc.any() and Fraction(int(y[acc].sum()), int(acc.sum())) <= a_goal:
            t_low, feasible = float(c), True
            break

    t_high = t_low
    for c in cands[::-1]:
        if c < t_low:
            break
        if Fraction(int(((p >= t_low) & (p < c)).sum()), n) <= e_goal:
            t_high = float(c)
            break

    acc, esc, flg = p < t_low, (p >= t_low) & (p < t_high), p >= t_high
    share = lambda m: float(m.mean())
    return Policy(
        t_low=t_low, t_high=t_high, feasible=feasible, n=n, n_distinct=len(cands),
        accept_rate=share(acc), accept_hallucinated_share=float(y[acc].mean()) if acc.any() else float("nan"),
        escalate_rate=share(esc), flag_rate=share(flg),
        flag_hallucinated_share=float(y[flg].mean()) if flg.any() else float("nan"),
        plateau_share_t_low=share(p == t_low), plateau_share_t_high=share(p == t_high),
    )


def summarize_ece(per_fold: pd.DataFrame, keys: Sequence[str], platt_margin: float) -> pd.DataFrame:
    """Mean ECE per method over folds and seeds, and the chosen method, per `keys` group."""
    rows = []
    for k, g in per_fold.groupby(list(keys), sort=False):
        means = g.groupby("method")["ece"].mean().to_dict()
        by_seed = g.groupby(["method", "seed"])["ece"].mean().unstack("method")
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                     "ece_platt": means["platt"], "ece_isotonic": means["isotonic"],
                     "ece_platt_seed_min": by_seed["platt"].min(), "ece_platt_seed_max": by_seed["platt"].max(),
                     "ece_isotonic_seed_min": by_seed["isotonic"].min(),
                     "ece_isotonic_seed_max": by_seed["isotonic"].max(),
                     "platt_minus_isotonic": means["platt"] - means["isotonic"],
                     "chosen_method": choose_method(means, platt_margin)})
    return pd.DataFrame(rows)
