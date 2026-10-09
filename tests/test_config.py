import pytest

from hdetect.config import ConfigError, load_config, parse_config
from tests.conftest import ROOT


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.resample_n == 5
    assert set(cfg.splits) == {"train", "val_id", "test_id", "test_ood", "test_heldout"}
    assert [s.name for s in cfg.splits_for("truthfulqa")] == ["train", "test_ood"]
    assert [s.name for s in cfg.splits_for("halueval")] == ["val_id", "test_id"]


def test_hash_is_stable_and_sensitive(raw_cfg):
    a = parse_config(raw_cfg).hash
    assert parse_config(raw_cfg).hash == a
    raw_cfg["seed"] += 1
    assert parse_config(raw_cfg).hash != a


def test_resample_n_is_preregistered(raw_cfg):
    raw_cfg["resample_n"] = 10
    with pytest.raises(ConfigError, match="pre-registered"):
        parse_config(raw_cfg)


def test_fractions_must_sum_to_one(raw_cfg):
    raw_cfg["splits"]["train"]["fraction"] = 0.8
    with pytest.raises(ConfigError, match="sum to 1"):
        parse_config(raw_cfg)


@pytest.mark.parametrize(
    "split, field, value",
    [
        ("test_heldout", "dataset", "halueval"),
        ("val_id", "generator", "model_b"),
        ("train", "generator", "model_b"),
    ],
)
def test_split_roles_are_enforced(raw_cfg, split, field, value):
    raw_cfg["splits"][split][field] = value
    with pytest.raises(ConfigError):
        parse_config(raw_cfg)


def test_unknown_split_rejected(raw_cfg):
    raw_cfg["splits"]["extra"] = {"dataset": "fever", "generator": "model_a", "n": 10}
    with pytest.raises(ConfigError, match="splits must be exactly"):
        parse_config(raw_cfg)


def test_single_split_dataset_cannot_use_fraction(raw_cfg):
    raw_cfg["splits"]["test_heldout"] = {"dataset": "fever", "generator": "model_b", "fraction": 0.5}
    with pytest.raises(ConfigError, match="`fraction` is for dividing"):
        parse_config(raw_cfg)


def test_fraction_and_n_cannot_mix_within_dataset(raw_cfg):
    raw_cfg["splits"]["val_id"] = {"dataset": "halueval", "generator": "model_a", "fraction": 0.5}
    with pytest.raises(ConfigError, match="`fraction` is for dividing"):
        parse_config(raw_cfg)


def test_several_sampled_splits_each_need_n(raw_cfg):
    del raw_cfg["splits"]["test_id"]["n"]
    with pytest.raises(ConfigError, match="each needs `n`"):
        parse_config(raw_cfg)


def test_split_takes_fraction_or_n_not_both(raw_cfg):
    raw_cfg["splits"]["test_id"]["fraction"] = 0.5
    with pytest.raises(ConfigError, match="not both"):
        parse_config(raw_cfg)


@pytest.mark.parametrize("threshold", [0, 1.5])
def test_dedup_threshold_bounds(raw_cfg, threshold):
    raw_cfg["dedup"]["threshold"] = threshold
    with pytest.raises(ConfigError):
        parse_config(raw_cfg)


def test_keep_priority_lists_every_dataset(raw_cfg):
    raw_cfg["dedup"]["keep_priority"] = ["fever", "halueval"]
    with pytest.raises(ConfigError):
        parse_config(raw_cfg)


def test_halueval_floor_below_threshold(raw_cfg):
    raw_cfg["labeling"]["halueval"]["unresolved_floor"] = 0.9
    with pytest.raises(ConfigError):
        parse_config(raw_cfg)


def test_missing_section_is_config_error(raw_cfg):
    del raw_cfg["audit"]
    with pytest.raises(ConfigError, match="malformed"):
        parse_config(raw_cfg)
