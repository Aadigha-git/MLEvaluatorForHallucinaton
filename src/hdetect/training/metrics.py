"""Thresholds, metrics, and the paired cluster bootstrap (docs/PREREGISTRATION.md, "Metrics").

Threshold rule: candidates are the unique observed scores; predict hallucinated when
score >= threshold; maximize hallucinated-class F1; ties go to the lowest threshold. For single
features the direction is chosen with the threshold: the score is oriented (s or -s) and ties go to
"higher means hallucinated", then to the lowest threshold on the oriented scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Threshold:
    value: float
    f1: float
    direction: str = "high"

    def orient(self, scores: np.ndarray) -> np.ndarray:
        return scores if self.direction == "high" else -scores

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (self.orient(np.asarray(scores, dtype=float)) >= self.value).astype(int)


def f1_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F1 at every unique score used as threshold (score >= t), thresholds ascending."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    thresholds = np.unique(scores)
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], y[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = tp[last][::-1], fp[last][::-1]
    fn = t.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0)
    return thresholds, f1


def best_threshold(scores: np.ndarray, y: np.ndarray) -> Threshold:
    thresholds, f1 = f1_curve(scores, y)
    i = int(np.argmax(f1))
    return Threshold(float(thresholds[i]), float(f1[i]))


def best_oriented_threshold(scores: np.ndarray, y: np.ndarray) -> Threshold:
    high = best_threshold(scores, y)
    low_t = best_threshold(-np.asarray(scores, dtype=float), y)
    low = Threshold(low_t.value, low_t.f1, "low")
    return low if low.f1 > high.f1 else high


def point_metrics(scores: np.ndarray, y: np.ndarray, threshold: Threshold) -> dict[str, float]:
    oriented = threshold.orient(np.asarray(scores, dtype=float))
    pred = threshold.predict(scores)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    two_class = len(np.unique(y)) == 2
    return {
        "threshold": threshold.value,
        "direction": threshold.direction,
        "f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "auroc": float(roc_auc_score(y, oriented)) if two_class else float("nan"),
        "auprc": float(average_precision_score(y, oriented)) if two_class else float("nan"),
        "predicted_positive_rate": float(pred.mean()),
    }


def all_hallucinated_f1(y: np.ndarray) -> float:
    p = float(np.mean(y))
    return 2 * p / (1 + p) if p > 0 else 0.0


# Weighted metrics over many bootstrap replicates at once. W is (replicates x rows) of
# non-negative row weights (cluster draw counts); scores and labels are fixed per evaluator.

def weighted_f1(W: np.ndarray, pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = W @ (pred * y)
    fp = W @ (pred * (1 - y))
    fn = W @ ((1 - pred) * y)
    denom = 2 * tp + fp + fn
    degenerate = denom == 0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return f1, degenerate


def weighted_ranking(W: np.ndarray, scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUROC and average precision (sklearn's definitions with sample weights); NaN for one-class replicates."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    s, t = scores[order], y[order].astype(float)
    Wo = W[:, order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(Wo * t, axis=1)[:, last]
    fp = np.cumsum(Wo * (1 - t), axis=1)[:, last]
    P, N = tp[:, -1:], fp[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.hstack([np.zeros_like(P), tp / P])
        fpr = np.hstack([np.zeros_like(N), fp / N])
        auroc = np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2, axis=1)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        ap = np.sum(np.diff(tpr, axis=1) * precision, axis=1)
    one_class = (P[:, 0] == 0) | (N[:, 0] == 0)
    auroc[one_class] = np.nan
    ap[one_class] = np.nan
    return auroc, ap


def cluster_weights(groups: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Row weights for `replicates` cluster bootstrap draws: every row of a drawn cluster is kept together."""
    clusters, index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=replicates)
    return counts[:, index].astype(float)


@dataclass(frozen=True)
class Evaluator:
    """One evaluator: per-seed scores (one entry for unseeded single features) and fixed thresholds."""

    name: str
    scores: Sequence[np.ndarray]
    thresholds: Sequence[Threshold]


def bootstrap(
    evaluators: Sequence[Evaluator],
    y: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
    chunk: int,
    pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, dict[str, float]]:
    """Paired cluster bootstrap of the seed-averaged F1, AUROC, and AUPRC; thresholds held fixed.

    Every evaluator (and the all-hallucinated baseline) is scored on the same resampled clusters.
    Returns 95% percentile intervals and degenerate-replicate counts, plus differences for `pairs`.
    """
    y = np.asarray(y, dtype=int)
    stats: dict[str, dict[str, list[np.ndarray]]] = {e.name: {"f1": [], "auroc": [], "auprc": []} for e in evaluators}
    stats["all_hallucinated"] = {"f1": []}
    degenerate: dict[str, list[np.ndarray]] = {e.name: [] for e in evaluators}
    rng_seed = seed
    done = 0
    while done < replicates:
        n = min(chunk, replicates - done)
        W = cluster_weights(groups, n, rng_seed + done)
        pos = W @ y
        p = pos / W.sum(axis=1)
        stats["all_hallucinated"]["f1"].append(2 * p / (1 + p))
        for e in evaluators:
            f1s, aurocs, aps, deg = [], [], [], np.zeros(n, dtype=bool)
            for s, th in zip(e.scores, e.thresholds):
                f1, d = weighted_f1(W, th.predict(s), y)
                auroc, ap = weighted_ranking(W, th.orient(np.asarray(s, dtype=float)), y)
                f1s.append(f1)
                aurocs.append(auroc)
                aps.append(ap)
                deg |= d
            stats[e.name]["f1"].append(np.mean(f1s, axis=0))
            stats[e.name]["auroc"].append(np.mean(aurocs, axis=0))
            stats[e.name]["auprc"].append(np.mean(aps, axis=0))
            degenerate[e.name].append(deg)
        done += n

    out: dict[str, dict[str, float]] = {}
    joined = {name: {m: np.concatenate(v) for m, v in ms.items()} for name, ms in stats.items()}
    deg_all = {name: np.concatenate(v) for name, v in degenerate.items()}
    for name, ms in joined.items():
        row: dict[str, float] = {}
        for m, values in ms.items():
            ok = values[~np.isnan(values)]
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
            row[f"{m}_skipped_replicates"] = int(np.isnan(values).sum())
        row["f1_degenerate_replicates"] = int(deg_all[name].sum()) if name in deg_all else 0
        out[name] = row
    for a, b in pairs:
        diff = out.setdefault(f"{a} - {b}", {})
        for m in ("f1", "auroc", "auprc"):
            if m not in joined[a] or m not in joined[b]:
                continue
            d = joined[a][m] - joined[b][m]
            ok = d[~np.isnan(d)]
            lo, hi = np.percentile(ok, [2.5, 97.5])
            diff.update({f"{m}_diff_ci_low": lo, f"{m}_diff_ci_high": hi, f"{m}_diff_mean": float(ok.mean()),
                         f"{m}_skipped_replicates": int(np.isnan(d).sum())})
        diff["f1_degenerate_replicates"] = int((deg_all.get(a, 0) | deg_all.get(b, 0)).sum()) \
            if a in deg_all or b in deg_all else 0
    return out
