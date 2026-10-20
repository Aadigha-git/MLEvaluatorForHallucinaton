"""Step 10: failure analysis and adversarial perturbation tests (exploratory; docs/SPEC.md build step 10).

Subcommands, all on the frozen Step 7 artifacts and Step 9 data loading:

- `export-failures`: the primary classifier's errors on the test splits (configs/step10.yaml), a
  seeded sample of up to 50 per error type, written for hand-sorting to a git-ignored CSV with the
  question, answer, label, references, retrieved passages, and feature values. Error counts only go
  to the report directory.
- `summarize-failures`: category counts from the hand-sorted CSV (no text) for the report.
- `perturb`: rule-based perturbations of every evaluated answer (src/hdetect/evaluation/perturb.py),
  features recomputed for the perturbed answer with the frozen feature models, and how often each
  verdict flips. The question, retrieved passages, and resamples are never changed.

`--stand-in` replaces every test split with val_id and writes to the git-ignored stand-in directory.
Without it, configs/step10.yaml must say `status: frozen`, and no report file is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import ROOT, Step9, git_state, sha256, subset_mask
from hdetect.cleaning import clean_answer
from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.evaluation.data import read_auto
from hdetect.evaluation.perturb import entity_pool, perturb
from hdetect.features.access import load_inputs
from hdetect.features.extract import usable_resamples
from hdetect.features.pipeline import build_extractor, load_feature_config
from hdetect.features.retrieval import passages_from_row
from hdetect.generation.config import load_generation_config
from hdetect.schema import Example
from hdetect.training import TIERS, tier_features
from hdetect.training.access import OPENED, data_access_table, install_audit_hook, is_test_path
from hdetect.training.calibration import calibrated
from hdetect.training.data import ACCESS_LOG, read_manifest
from hdetect.training.metrics import cluster_weights
from hdetect.training.models import score
from hdetect.training.run import ROUND

FEATURES_SHOWN = ("lex_overlap_max", "sim_max", "nli_entail_max", "nli_contra_max", "hedge_rate", "log_length",
                  "specificity", "self_contra", "sc_embed", "sc_entail", "sc_contra")


class Context(Step9):
    """Step 9's frozen inputs and test-data loading, without its report-directory checks."""

    def __init__(self, stand_in: bool, s10: dict) -> None:
        self.args = argparse.Namespace(config="configs/default.yaml", feature_config="configs/features.yaml",
                                       training_config="configs/training.yaml",
                                       generation_config="configs/generation.yaml",
                                       evaluation_config="configs/evaluation.yaml", stand_in=stand_in)
        self.stand_in = stand_in
        self.ecfg = yaml.safe_load((ROOT / self.args.evaluation_config).read_text())
        self.cfg = load_config(ROOT / self.args.config)
        self.fcfg = load_feature_config(ROOT / self.args.feature_config)
        self.tcfg = yaml.safe_load((ROOT / self.args.training_config).read_text())
        self.gcfg = load_generation_config(ROOT / self.args.generation_config)
        self.splits = list(s10["splits"])
        self.tables, self.timings = {}, []
        self.load_frozen()
        self.load_test()

    def examples(self) -> dict[str, Example]:
        path = Path(self.cfg.paths["processed"]) / "examples.jsonl"
        ACCESS_LOG.append(path.as_posix())
        return {e.question_id: e for e in map(Example.from_dict, read_jsonl(path))}

    def queries(self, split: str) -> dict[str, str]:
        src = self.source(split)
        manifest = read_manifest(Path(self.cfg.paths["splits"]) / "split_manifest.jsonl")
        examples = Path(self.cfg.paths["processed"]) / "examples.jsonl"
        ACCESS_LOG.append(examples.as_posix())
        return {q.question_id: q.query for q in load_inputs(src, examples, manifest, allow_test_splits=True)}

    def passages(self, split: str) -> dict[str, list[str]]:
        path = Path(self.fcfg.paths["features"]) / "retrieval" / f"{self.source(split)}.jsonl"
        ACCESS_LOG.append(path.as_posix())
        return {r["question_id"]: [p.formatted() for p in passages_from_row(r)] for r in read_jsonl(path)}


def majority(preds: list[np.ndarray]) -> np.ndarray:
    return (np.sum(preds, axis=0) * 2 > len(preds)).astype(int)


def allocate(available: dict[str, int], total: int, order: list[str]) -> dict[str, int]:
    """Round-robin over `order`: equal shares, remainder to earlier splits, shortfalls filled by the rest."""
    quota = {s: 0 for s in order}
    remaining = min(total, sum(available.values()))
    while remaining:
        for s in order:
            if remaining and quota[s] < available[s]:
                quota[s] += 1
                remaining -= 1
    return quota


def write_new(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        sys.exit(f"{path} exists; not overwritten")
    df.round(ROUND).to_csv(path, index=False)


# Failure analysis

def errors(ctx: Context, fa: dict) -> pd.DataFrame:
    tier, cond = fa["tier"], fa["condition"]
    rows = []
    for split in ctx.splits:
        d = ctx.data[split][cond]
        m = subset_mask(d, tier, "all")
        pred = majority(ctx.classifier(tier, cond, d, m).predictions())
        y = d.y[m]
        art = ctx.artifacts[(tier, cond)][0]
        p = calibrated(art["calibrator"], d.subset(tuple(art["features"]))[m])
        meta, X = d.meta[m].reset_index(drop=True), d.X[m].reset_index(drop=True)
        for i in np.flatnonzero(pred != y):
            rows.append({"split": split, "answer_id": meta.at[i, "answer_id"], "question_id": meta.at[i, "question_id"],
                         "error_type": "false_positive" if pred[i] == 1 else "false_negative",
                         "label": "hallucinated" if y[i] == 1 else "not_hallucinated",
                         "calibrated_p_seed20261008": float(p[i]),
                         **{f: float(X.at[i, f]) for f in FEATURES_SHOWN if f in X.columns}})
    return pd.DataFrame(rows)


def sample_errors(err: pd.DataFrame, fa: dict, order: list[str]) -> pd.DataFrame:
    picked = []
    for etype in ("false_positive", "false_negative"):
        own = err[err["error_type"] == etype]
        avail = {s: int((own["split"] == s).sum()) for s in order}
        for s, k in allocate(avail, fa["max_per_error_type"], order).items():
            ids = sorted(own.loc[own["split"] == s, "answer_id"])
            picked += [(s, a) for a in random.Random(f"{fa['sample_seed']}:{etype}:{s}").sample(ids, k)]
    keep = [(s, a) in set(picked) for s, a in zip(err["split"], err["answer_id"])]
    return err[keep].sort_values(["error_type", "split", "answer_id"]).reset_index(drop=True)


def export_failures(ctx: Context, s10: dict, out: Path) -> None:
    fa = s10["failure_analysis"]
    queue_path = out / "failures_stand_in.csv" if ctx.stand_in else ROOT / fa["queue"]
    if queue_path.exists() and not ctx.stand_in:
        sys.exit(f"{queue_path} exists; hand-sorted categories are never overwritten")
    err = errors(ctx, fa)
    sample = sample_errors(err, fa, ctx.splits)
    counts = (err.groupby(["split", "error_type"]).size().rename("errors").reset_index()
              .merge(sample.groupby(["split", "error_type"]).size().rename("sampled").reset_index(),
                     how="left", on=["split", "error_type"]).fillna({"sampled": 0}))
    counts["tier"], counts["condition"] = fa["tier"], fa["condition"]
    examples = ctx.examples()
    rows = []
    for split in ctx.splits:
        own = sample[sample["split"] == split]
        if own.empty:
            continue
        auto = {r["answer_id"]: r for r in read_auto(ctx.tcfg["paths"]["labels"], ctx.source(split), allow_test_splits=True)}
        queries, passages, answers = ctx.queries(split), ctx.passages(split), ctx.answer_rows[split]
        for r in own.to_dict("records"):
            qid, a = r["question_id"], auto[r["answer_id"]]
            ex = examples[qid]
            rows.append({"item": f"{r['error_type'][:2].upper()}-{len(rows) + 1:03d}", **{k: r[k] for k in (
                "split", "error_type", "answer_id")}, "dataset": a["dataset"], "generator": a["generator"],
                "question": queries[qid], "answer": clean_answer(answers[qid].primary["text"]), "label": r["label"],
                "label_source": "hand" if a["label"] == "unresolved" else "auto", "auto_reason": a["reason"],
                "correct_refs": " | ".join(ex.correct_refs), "incorrect_refs": " | ".join(ex.incorrect_refs),
                "gold_label": ex.gold_label or "", "gold_evidence": " | ".join(ex.gold_evidence),
                "retrieved_passages": "\n---\n".join(passages[qid]),
                **{k: r[k] for k in r if k.startswith("calibrated") or k in FEATURES_SHOWN},
                "category": "", "notes": ""})
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_new(out / "failure_counts.csv", counts, ctx.stand_in)
    print(f"failure queue: {len(rows)} answers -> {queue_path.relative_to(ROOT)}")


def summarize_failures(ctx: Context | None, s10: dict, out: Path, stand_in: bool) -> None:
    fa = s10["failure_analysis"]
    queue_path = out / "failures_stand_in.csv" if stand_in else ROOT / fa["queue"]
    df = pd.read_csv(queue_path, keep_default_na=False)
    bad = df[~df["category"].isin(fa["categories"])]
    if len(bad):
        sys.exit(f"{len(bad)} rows have no category or one outside {fa['categories']}")
    keys = ["error_type", "split", "generator", "category"]
    counts = df.groupby(keys).size().rename("n").reset_index()
    totals = df.groupby(["error_type", "category"]).size().rename("n").reset_index().assign(split="all", generator="all")
    write_new(out / "failure_categories.csv", pd.concat([counts, totals[keys + ["n"]]], ignore_index=True), stand_in)
    print(f"wrote {(out / 'failure_categories.csv').relative_to(ROOT)}")


# Perturbations

def perturbed_features(ctx: Context, split: str, pc: dict, extractor) -> tuple[dict, pd.DataFrame]:
    """Per perturbation: a feature frame aligned with the split's rows (NaN where it does not apply)."""
    d = ctx.data[split]["retrieved"]
    queries, passages, answers = ctx.queries(split), ctx.passages(split), ctx.answer_rows[split]
    qids = list(d.meta["question_id"])
    texts = {q: clean_answer(answers[q].primary["text"]) for q in qids}
    pool = entity_pool((texts[q], queries[q]) for q in qids)
    out = {}
    for name in pc["perturbations"]:
        rows = []
        t0 = time.perf_counter()
        for q, aid in zip(qids, d.meta["answer_id"]):
            new = perturb(name, texts[q], queries[q], pool, aid, pc["entity_seed"])
            if new is None:
                rows.append({"applicable": False})
                continue
            free = extractor.free(new, usable_resamples(answers[q]))["features"]
            ev = extractor.evidence(new, queries[q], passages[q])["features"]
            rows.append({"applicable": True, **free, **ev})
        out[name] = pd.DataFrame(rows).astype({"applicable": bool})
        print(f"  {split} {name}: {int(out[name]['applicable'].sum())} applicable, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    check = []
    for q in qids[:pc["recompute_check_per_split"]]:
        i = qids.index(q)
        free = extractor.free(texts[q], usable_resamples(answers[q]))["features"]
        ev = extractor.evidence(texts[q], queries[q], passages[q])["features"]
        for f, v in {**free, **ev}.items():
            if v is not None:
                check.append({"split": split, "feature": f, "abs_diff": abs(v - float(d.X.at[i, f]))})
    return out, pd.DataFrame(check)


def flip_stats(before: list[np.ndarray], after: list[np.ndarray], groups: np.ndarray, pc: dict, seed: int) -> dict:
    """Seed-averaged flip rates, with a cluster-bootstrap interval for the overall flip rate."""
    b, a = np.array(before), np.array(after)
    flip = (b != a).mean(axis=0)
    bt = pc["bootstrap"]
    rates, done = [], 0
    while done < bt["replicates"]:
        n = min(bt["chunk"], bt["replicates"] - done)
        W = cluster_weights(groups, n, seed + done)
        rates.append((W @ flip) / W.sum(axis=1))
        done += n
    lo, hi = np.percentile(np.concatenate(rates), [2.5, 97.5])
    return {"flip_rate": float(flip.mean()), "flip_ci_low": float(lo), "flip_ci_high": float(hi),
            "to_hallucinated_rate": float(((b == 0) & (a == 1)).mean()),
            "to_not_hallucinated_rate": float(((b == 1) & (a == 0)).mean()),
            "predicted_hallucinated_before": float(b.mean()), "predicted_hallucinated_after": float(a.mean())}


def run_perturb(ctx: Context, s10: dict, out: Path) -> None:
    pc = s10["perturbation"]
    cache = ROOT / pc["cache"]
    extractor = build_extractor(ctx.fcfg, cache)
    bt = pc["bootstrap"]
    rows, checks = [], []
    for k, split in enumerate(ctx.splits):
        feats, check = perturbed_features(ctx, split, pc, extractor)
        checks.append(check)
        for name, pf in feats.items():
            for cond in pc["conditions"]:
                d = ctx.data[split][cond]
                for tier in TIERS:
                    cols = tier_features(tier, cond)
                    m = subset_mask(d, tier, "all") & pf["applicable"].to_numpy()
                    if tier == "resampling":
                        m &= pf["sc_embed"].notna().to_numpy() if "sc_embed" in pf else False
                    keys = {"split": split, "condition": cond, "tier": tier, "perturbation": name,
                            "n_rows": int(subset_mask(d, tier, "all").sum()), "n_applicable": int(m.sum())}
                    if not m.any():
                        rows.append(keys)
                        continue
                    Xb = d.X.loc[m, list(cols)].reset_index(drop=True)
                    Xa = pf.loc[m, list(cols)].astype(float).reset_index(drop=True)
                    arts = ctx.artifacts[(tier, cond)]
                    ths = [a["f1_threshold"]["value"] for a in arts]
                    sb = [score(a["estimator"], np.ascontiguousarray(Xb[a["features"]].to_numpy())) for a in arts]
                    sa = [score(a["estimator"], np.ascontiguousarray(Xa[a["features"]].to_numpy())) for a in arts]
                    groups = d.groups[m]
                    seed = bt["seed"] + k * bt["seed_stride"]
                    stats = flip_stats([(s >= t).astype(int) for s, t in zip(sb, ths)],
                                       [(s >= t).astype(int) for s, t in zip(sa, ths)], groups, pc, seed)
                    pol = arts[0]["policy"]
                    band = lambda X: np.digitize(calibrated(arts[0]["calibrator"],
                                                            np.ascontiguousarray(X[arts[0]["features"]].to_numpy())),
                                                 [pol["t_low"], pol["t_high"]])
                    rows.append({**keys, "evaluator": "classifier", **stats,
                                 "mean_score_change": float(np.mean([np.mean(a - b) for a, b in zip(sa, sb)])),
                                 "band_change_rate_seed20261008": float((band(Xb) != band(Xa)).mean())})
                    feat, th = ctx.single[(tier, cond)]
                    fb, fa_ = th.predict(Xb[feat].to_numpy()), th.predict(Xa[feat].to_numpy())
                    rows.append({**keys, "evaluator": f"feature:{feat}",
                                 **flip_stats([fb], [fa_], groups, pc, seed),
                                 "mean_score_change": float(np.mean(Xa[feat].to_numpy() - Xb[feat].to_numpy()))})
    write_new(out / "perturbation_flips.csv", pd.DataFrame(rows), ctx.stand_in)
    check = pd.concat(checks)
    summary = check.groupby(["split", "feature"])["abs_diff"].agg(["count", "max", "mean"]).reset_index()
    write_new(out / "perturbation_recompute_check.csv", summary, ctx.stand_in)
    print(f"wrote {(out / 'perturbation_flips.csv').relative_to(ROOT)}")


def finish(ctx: Context, out: Path, command: str, s10_path: Path) -> None:
    access = data_access_table(ROOT)
    if ctx.stand_in and (access["test_split"].any() or any(is_test_path(p) for p in OPENED + ACCESS_LOG)):
        sys.exit("stand-in run touched a test-split path")
    access.to_csv(out / f"data_access_{command}.csv", index=False)
    manifest = {"command": command, "stand_in": ctx.stand_in, "git": git_state(),
                "step10_config_sha256": sha256(s10_path),
                "inputs_sha256": {p: sha256(ROOT / p) for p in access["path"] if (ROOT / p).is_file()
                                  and not p.endswith(".sqlite")}}
    (out / f"manifest_{command}.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["export-failures", "summarize-failures", "perturb"])
    p.add_argument("--step10-config", default="configs/step10.yaml")
    p.add_argument("--stand-in", action="store_true", help="dry run with val_id in place of every test split")
    p.add_argument("--replicates", type=int, help="stand-in only: fewer bootstrap replicates")
    args = p.parse_args()
    s10_path = ROOT / args.step10_config
    s10 = yaml.safe_load(s10_path.read_text())
    if not args.stand_in and s10["status"] != "frozen":
        sys.exit("configs/step10.yaml is provisional: only `--stand-in` may run")
    if args.replicates:
        if not args.stand_in:
            sys.exit("--replicates is only for the stand-in dry run")
        s10["perturbation"]["bootstrap"]["replicates"] = args.replicates
    if args.stand_in:
        s10["perturbation"]["cache"] = s10["perturbation"]["cache"].replace(".sqlite", "_stand_in.sqlite")
    out = ROOT / s10["paths"]["stand_in_reports" if args.stand_in else "reports"]
    out.mkdir(parents=True, exist_ok=True)
    install_audit_hook()
    if args.command == "summarize-failures":
        summarize_failures(None, s10, out, args.stand_in)
        return
    ctx = Context(args.stand_in, s10)
    {"export-failures": export_failures, "perturb": run_perturb}[args.command](ctx, s10, out)
    finish(ctx, out, args.command, s10_path)


if __name__ == "__main__":
    main()
