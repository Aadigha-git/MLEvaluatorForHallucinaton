"""Bootstrap replicates and the pre-registered RQ1 and RQ2 rules (docs/PREREGISTRATION.md).

Every statistic is seed-averaged within each replicate: computed per training seed, then averaged.
Within a split, every evaluator is scored on the same resampled clusters, because the cluster draws
depend only on the split's cluster IDs and its bootstrap seed. Across splits (the RQ2 drop), each
split is resampled independently with its own seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hdetect.training.metrics import Threshold, cluster_weights, weighted_f1, weighted_ranking

YES, MIXED, NO = "yes", "mixed", "no"
# Seed-averaged statistics are rounded here before any comparison, so that evaluators making the
# same predictions differ by exactly 0 rather than by floating-point noise of either sign.
DECIMALS = 12


@dataclass(frozen=True)
class Scored:
    """One evaluator on one split: per-seed scores and their fixed thresholds (one entry if unseeded).

    `ranking=False` marks fixed binary predictions (an LLM judge), for which AUROC is not defined.
    """

    name: str
    scores: Sequence[np.ndarray]
    thresholds: Sequence[Threshold]
    ranking: bool = True

    def predictions(self) -> list[np.ndarray]:
        return [th.predict(s) for s, th in zip(self.scores, self.thresholds)]


def fixed_predictions(name: str, pred: np.ndarray) -> Scored:
    return Scored(name, [np.asarray(pred, dtype=float)], [Threshold(0.5, float("nan"))], ranking=False)


def judge_predictions(verdicts: Sequence[int | None], y: np.ndarray) -> np.ndarray:
    """A missing verdict (API or parse failure) is scored as wrong (Step 8 amendment)."""
    return np.array([(1 - t) if v is None else v for v, t in zip(verdicts, y)], dtype=float)


def point_f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def all_hallucinated_f1(y: np.ndarray) -> float:
    """2p / (1 + p), computed with the same arithmetic as every other F1 so that a classifier that
    flags everything differs from it by exactly 0, not by rounding noise of either sign."""
    return round(point_f1(np.ones(len(y)), np.asarray(y)), DECIMALS)


def seed_point(e: Scored, y: np.ndarray) -> dict[str, float]:
    """Seed-averaged point F1 and AUROC, with the spread over seeds."""
    from sklearn.metrics import roc_auc_score

    f1s = [point_f1(p, y) for p in e.predictions()]
    out = {"f1": round(float(np.mean(f1s)), DECIMALS), "f1_seed_min": float(np.min(f1s)), "f1_seed_max": float(np.max(f1s)),
           "f1_seed_std": float(np.std(f1s)), "n_seeds": len(f1s)}
    if e.ranking and len(np.unique(y)) == 2:
        aucs = [roc_auc_score(y, th.orient(np.asarray(s, dtype=float))) for s, th in zip(e.scores, e.thresholds)]
        out.update(auroc=round(float(np.mean(aucs)), DECIMALS), auroc_seed_min=float(np.min(aucs)), auroc_seed_max=float(np.max(aucs)),
                   auroc_seed_std=float(np.std(aucs)))
    else:
        out.update(auroc=np.nan, auroc_seed_min=np.nan, auroc_seed_max=np.nan, auroc_seed_std=np.nan)
    return out


@dataclass
class Replicates:
    """Seed-averaged statistics per bootstrap replicate for several evaluators on one split."""

    f1: dict[str, np.ndarray]
    auroc: dict[str, np.ndarray]
    degenerate: dict[str, np.ndarray]
    all_hallucinated: np.ndarray


def replicate(evaluators: Sequence[Scored], y: np.ndarray, groups: np.ndarray, replicates: int, seed: int,
              chunk: int) -> Replicates:
    y = np.asarray(y, dtype=int)
    f1 = {e.name: [] for e in evaluators}
    auc = {e.name: [] for e in evaluators}
    deg = {e.name: [] for e in evaluators}
    base = []
    done = 0
    while done < replicates:
        n = min(chunk, replicates - done)
        W = cluster_weights(groups, n, seed + done)
        base.append(np.round(weighted_f1(W, np.ones(len(y)), y)[0], DECIMALS))
        for e in evaluators:
            f1s, aucs, d = [], [], np.zeros(n, dtype=bool)
            for s, th in zip(e.scores, e.thresholds):
                f, dd = weighted_f1(W, th.predict(s), y)
                f1s.append(f)
                d |= dd
                if e.ranking:
                    aucs.append(weighted_ranking(W, th.orient(np.asarray(s, dtype=float)), y)[0])
            f1[e.name].append(np.round(np.mean(f1s, axis=0), DECIMALS))
            auc[e.name].append(np.round(np.mean(aucs, axis=0), DECIMALS) if aucs else np.full(n, np.nan))
            deg[e.name].append(d)
        done += n
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return Replicates(cat(f1), cat(auc), cat(deg), np.concatenate(base))


def interval(values: np.ndarray) -> dict[str, float]:
    ok = values[~np.isnan(values)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
    return {"ci_low": float(lo), "ci_high": float(hi), "skipped_replicates": int(np.isnan(values).sum())}


def rq1_split(rep: Replicates, point: dict[str, dict], clf: str, single: str, metric: str = "f1") -> dict:
    """D = metric(classifier) - metric(best single feature), paired within the split."""
    a, b = getattr(rep, metric)[clf], getattr(rep, metric)[single]
    ci = interval(np.round(a - b, DECIMALS))
    return {"metric": metric, "classifier_value": point[clf][metric], "best_single_value": point[single][metric],
            "diff": round(point[clf][metric] - point[single][metric], DECIMALS), "diff_ci_low": ci["ci_low"],
            "diff_ci_high": ci["ci_high"], "skipped_replicates": ci["skipped_replicates"],
            "degenerate_replicates": int((rep.degenerate[clf] | rep.degenerate[single]).sum()) if metric == "f1" else 0,
            "passes": bool(ci["ci_low"] > 0)}


def rq_outcome(passes: Sequence[bool]) -> str:
    """Yes if every split passes, mixed if exactly one of two does, no if none does."""
    k = sum(bool(p) for p in passes)
    return YES if k == len(passes) else (NO if k == 0 else MIXED)


def rq2_split(ref: Replicates, rep: Replicates, ref_point: dict, point: dict, clf: str, margin: float,
              y: np.ndarray, metric: str = "f1") -> dict:
    """Drop from the reference split to S (independent resampling), and S against the trivial baseline.

    For F1 the trivial baseline is predicting every answer hallucinated (2p / (1 + p), p within each
    replicate, via `weighted_f1` on all-ones predictions); for the AUROC version it is chance, 0.5.
    """
    drop = np.round(getattr(ref, metric)[clf] - getattr(rep, metric)[clf], DECIMALS)
    if metric == "f1":
        vs = np.round(rep.f1[clf] - rep.all_hallucinated, DECIMALS)
        baseline_point = all_hallucinated_f1(y)
    else:
        vs = np.round(rep.auroc[clf] - 0.5, DECIMALS)
        baseline_point = 0.5
    d_ci, v_ci = interval(drop), interval(vs)
    delta = round(ref_point[metric] - point[metric], DECIMALS)
    return {"metric": metric, "reference_value": ref_point[metric], "split_value": point[metric], "drop": delta,
            "drop_ci_low": d_ci["ci_low"], "drop_ci_high": d_ci["ci_high"], "drop_within_margin": bool(delta <= margin),
            "trivial_baseline": baseline_point, "vs_trivial": round(point[metric] - baseline_point, DECIMALS),
            "vs_trivial_ci_low": v_ci["ci_low"], "vs_trivial_ci_high": v_ci["ci_high"],
            "beats_trivial": bool(v_ci["ci_low"] > 0),
            "skipped_replicates": d_ci["skipped_replicates"] + v_ci["skipped_replicates"],
            "degenerate_replicates": int(rep.degenerate[clf].sum() + ref.degenerate[clf].sum()) if metric == "f1" else 0,
            "passes": bool(delta <= margin and v_ci["ci_low"] > 0)}


def band_rates(P: np.ndarray, y: np.ndarray, t_low: float, t_high: float) -> dict[str, float]:
    """Achieved accept/escalate/flag rates of a frozen policy on new data (exploratory drift)."""
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    acc, esc, flg = P < t_low, (P >= t_low) & (P < t_high), P >= t_high
    share = lambda m: float(m.mean()) if len(m) else np.nan
    return {"n": len(P), "accept_rate": share(acc), "escalate_rate": share(esc), "flag_rate": share(flg),
            "accept_hallucinated_share": float(y[acc].mean()) if acc.any() else np.nan,
            "flag_hallucinated_share": float(y[flg].mean()) if flg.any() else np.nan,
            "plateau_share_t_low": share(P == t_low), "plateau_share_t_high": share(P == t_high)}
