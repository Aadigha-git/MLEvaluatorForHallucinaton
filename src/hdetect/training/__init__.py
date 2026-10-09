"""Step 6: fit classifiers on train, select and report on val_id. Test splits are never read here.

Tiers are explicit: the single-shot tier never sees the resampling features.
"""

from hdetect.features import EVIDENCE_FEATURES, EVIDENCE_FREE_FEATURES, RESAMPLING_FEATURES

TIERS = ("single_shot", "resampling")
TRAIN_CONDITIONS = ("retrieved", "none")
EVAL_CONDITIONS = ("retrieved", "none", "gold")
FAMILIES = ("logistic_regression", "hist_gradient_boosting")

GROUPS = {
    "evidence_dependent": EVIDENCE_FEATURES,
    "evidence_free": EVIDENCE_FREE_FEATURES,
    "self_consistency": RESAMPLING_FEATURES,
}


def tier_features(tier: str, condition: str) -> tuple[str, ...]:
    """The features a classifier may use in this tier and evidence condition."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if condition not in EVAL_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    evidence = EVIDENCE_FEATURES if condition in ("retrieved", "gold") else ()
    resampling = RESAMPLING_FEATURES if tier == "resampling" else ()
    return evidence + EVIDENCE_FREE_FEATURES + resampling


def training_condition(condition: str) -> str:
    """The gold-evidence analysis reuses the retrieved-evidence classifier (train has no gold evidence)."""
    return "retrieved" if condition == "gold" else condition


def present_groups(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {name: tuple(f for f in feats if f in features) for name, feats in GROUPS.items() if set(feats) & set(features)}
