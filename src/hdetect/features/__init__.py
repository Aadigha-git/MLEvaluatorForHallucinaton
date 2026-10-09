"""Black-box features (docs/PREREGISTRATION.md, "Features"). Feature code never sees labels.

Groups are explicit: evidence-dependent features need passages; evidence-free features use the
primary answer only; resampling features use the 5 cached resamples (resampling tier only).
"""

EVIDENCE_FEATURES = ("sim_max", "nli_entail_max", "nli_contra_max", "lex_overlap_max")
EVIDENCE_FREE_FEATURES = ("hedge_rate", "booster_rate", "log_length", "specificity", "self_contra")
RESAMPLING_FEATURES = ("sc_embed", "sc_entail", "sc_contra")

CONDITIONS = ("retrieved", "none", "gold")
GOLD_DATASETS = frozenset({"halueval", "fever"})
TEST_SPLITS = frozenset({"test_id", "test_ood", "test_heldout"})
DEV_SPLITS = ("train", "val_id")


def feature_columns(condition: str) -> tuple[str, ...]:
    """Single-shot features first, then the resampling-tier features."""
    evidence = EVIDENCE_FEATURES if condition in ("retrieved", "gold") else ()
    return evidence + EVIDENCE_FREE_FEATURES + RESAMPLING_FEATURES
