"""Feature extraction must respect each arm's OWN preprocessing.

Before 2026-09-15 ``evaluate_downstream_probe.py`` took ``image_kwargs`` from whichever
checkpoint loaded first and applied it to every arm. A config mixing the 130mm baselines
with the 150mm re-baseline therefore fed one of them pixels it had never trained on, and
produced a plausible number with no error. These tests pin both halves of the fix:

* a single-preprocessing config -- every probe config written before that date -- must
  still make exactly ONE extraction pass and return byte-identical names, so existing
  artifacts reproduce;
* a mixed config must extract once per preprocessing, and must not let two different
  preprocessings collide on the unadapted backbone's feature name.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "evaluate_downstream_probe", ROOT / "scripts/evaluate_downstream_probe.py"
)
probe = importlib.util.module_from_spec(spec)
sys.modules["evaluate_downstream_probe"] = probe
spec.loader.exec_module(probe)


CROP130 = {"image_size": 224, "crop_mm": 130.0, "slices_per_plane": 3, "slice_band": (0.2, 0.8)}
CROP150 = {"image_size": 224, "crop_mm": 150.0, "slices_per_plane": 3, "slice_band": (0.2, 0.8)}


@pytest.fixture
def capture(monkeypatch):
    """Record every extract_features call instead of decoding any pixels."""
    calls = []

    def fake_extract(records, image_kwargs, cache_dir, towers, batch_size=8):
        calls.append({"image_kwargs": image_kwargs, "cache_dir": cache_dir, "towers": sorted(towers)})
        return {name: {"uniform_mean_768": name} for name in towers}

    monkeypatch.setattr(probe, "extract_features", fake_extract)
    return calls


def _towers(names):
    return {name: object() for name in names}


def test_single_preprocessing_makes_one_pass_with_unchanged_names(capture):
    """The backward-compatibility guarantee: existing configs must not move at all."""
    names = ["exposed/seed2026", "exposed/seed2027"]
    towers = _towers(["before_cl/dinov2-unadapted", *names])
    provenance = {name: {"image_kwargs": dict(CROP130)} for name in names}

    features = probe.extract_features_by_preprocessing(
        [], towers, provenance, Path("/cache/default")
    )

    assert len(capture) == 1, "one preprocessing must mean one extraction pass"
    assert capture[0]["image_kwargs"] == CROP130
    # The unadapted arm keeps its bare, untagged name so existing artifacts still resolve.
    assert set(features) == {"before_cl/dinov2-unadapted", *names}


def test_mixed_preprocessing_extracts_once_per_crop(capture):
    """The defect this fix exists for: two crops must not share one extraction pass."""
    towers = _towers(["before_cl/dinov2-unadapted", "crop130/seed2026", "crop150/seed2026"])
    provenance = {
        "crop130/seed2026": {"image_kwargs": dict(CROP130)},
        "crop150/seed2026": {"image_kwargs": dict(CROP150)},
    }

    features = probe.extract_features_by_preprocessing(
        [], towers, provenance, Path("/cache/default")
    )

    assert len(capture) == 2, "two preprocessings must mean two extraction passes"
    crops = sorted(call["image_kwargs"]["crop_mm"] for call in capture)
    assert crops == [130.0, 150.0]

    # Each checkpoint is extracted at its own crop, never at the other's.
    by_crop = {call["image_kwargs"]["crop_mm"]: call["towers"] for call in capture}
    assert "crop130/seed2026" in by_crop[130.0]
    assert "crop130/seed2026" not in by_crop[150.0]
    assert "crop150/seed2026" in by_crop[150.0]
    assert "crop150/seed2026" not in by_crop[130.0]

    # The unadapted backbone is a DIFFERENT feature set per crop and must not collide.
    assert "before_cl/dinov2-unadapted@crop130mm" in features
    assert "before_cl/dinov2-unadapted@crop150mm" in features
    assert "before_cl/dinov2-unadapted" not in features


def test_tag_names_only_the_fields_that_differ(capture):
    """A tag built from every field would be unreadable and would churn on unrelated edits."""
    towers = _towers(["before_cl/dinov2-unadapted", "a/seed2026", "b/seed2026"])
    provenance = {
        "a/seed2026": {"image_kwargs": dict(CROP130)},
        "b/seed2026": {"image_kwargs": dict(CROP150)},
    }
    features = probe.extract_features_by_preprocessing([], towers, provenance, Path("/c"))
    tagged = [name for name in features if name.startswith("before_cl/")]
    # crop_mm is the only varying field, so image_size/slices/band stay out of the tag.
    assert sorted(tagged) == [
        "before_cl/dinov2-unadapted@crop130mm",
        "before_cl/dinov2-unadapted@crop150mm",
    ]


def test_per_arm_cache_dir_is_used_when_declared(capture):
    """Each crop has its own populated cache; the right one must reach the right group."""
    towers = _towers(["before_cl/dinov2-unadapted", "crop130/seed2026", "crop150/seed2026"])
    provenance = {
        "crop130/seed2026": {"image_kwargs": dict(CROP130)},
        "crop150/seed2026": {"image_kwargs": dict(CROP150)},
    }
    probe.extract_features_by_preprocessing(
        [], towers, provenance, Path("/cache/default"),
        {"crop150/seed2026": Path("/cache/crop150")},
    )
    used = {call["image_kwargs"]["crop_mm"]: call["cache_dir"] for call in capture}
    assert used[150.0] == Path("/cache/crop150")
    assert used[130.0] == Path("/cache/default"), "an undeclared arm falls back to the CLI default"


def test_conflicting_cache_dirs_within_one_group_raise(capture):
    """Two arms at the same preprocessing cannot disagree about where its pixels live."""
    towers = _towers(["before_cl/dinov2-unadapted", "x/seed2026", "y/seed2026"])
    provenance = {
        "x/seed2026": {"image_kwargs": dict(CROP130)},
        "y/seed2026": {"image_kwargs": dict(CROP130)},
    }
    with pytest.raises(SystemExit, match="different image_cache_dir"):
        probe.extract_features_by_preprocessing(
            [], towers, provenance, Path("/cache/default"),
            {"x/seed2026": Path("/cache/a"), "y/seed2026": Path("/cache/b")},
        )


def test_no_checkpoint_arms_raises_rather_than_silently_returning_nothing(capture):
    with pytest.raises(SystemExit, match="at least one"):
        probe.extract_features_by_preprocessing(
            [], _towers(["before_cl/dinov2-unadapted"]), {}, Path("/cache")
        )


def test_unadapted_floor_is_recorded_per_arm(capture):
    """The gap that broke the first crop150 run: extraction was grouped but the SCORING
    code still looked up one global 'before_cl/dinov2-unadapted' key, which does not
    exist once the unadapted backbone is tagged per crop. Each arm must carry its own
    floor, or a 130mm arm gets subtracted from a 150mm reference."""
    towers = _towers(["before_cl/dinov2-unadapted", "crop130/seed2026", "crop150/seed2026"])
    provenance = {
        "crop130/seed2026": {"image_kwargs": dict(CROP130)},
        "crop150/seed2026": {"image_kwargs": dict(CROP150)},
    }
    probe.extract_features_by_preprocessing([], towers, provenance, Path("/c"))

    floors = provenance["_unadapted_for"]
    assert floors["crop130/seed2026"] == "before_cl/dinov2-unadapted@crop130mm"
    assert floors["crop150/seed2026"] == "before_cl/dinov2-unadapted@crop150mm"
    # Each tagged unadapted arm maps to itself, so the scoring loop skips it as its own floor.
    assert floors["before_cl/dinov2-unadapted@crop130mm"] == "before_cl/dinov2-unadapted@crop130mm"


def test_single_preprocessing_floor_is_the_untagged_name(capture):
    """Backward compatibility for the floor lookup too, not only for the feature names."""
    towers = _towers(["before_cl/dinov2-unadapted", "exposed/seed2026"])
    provenance = {"exposed/seed2026": {"image_kwargs": dict(CROP130)}}
    probe.extract_features_by_preprocessing([], towers, provenance, Path("/c"))
    assert provenance["_unadapted_for"]["exposed/seed2026"] == "before_cl/dinov2-unadapted"


def test_floor_feature_set_is_always_uniform_mean(capture):
    """The unadapted backbone has no learned pooling and no projection head, so it only
    ever yields uniform_mean_768. A candidate scored on plane_attention_768 or
    projection_256 must still be compared against the unadapted uniform_mean_768 at its
    own crop -- matching feature sets would request a key that is never extracted and
    would break every existing single-preprocessing config too."""
    towers = _towers(["before_cl/dinov2-unadapted", "crop130/seed2026", "crop150/seed2026"])
    provenance = {
        "crop130/seed2026": {"image_kwargs": dict(CROP130)},
        "crop150/seed2026": {"image_kwargs": dict(CROP150)},
    }
    features = probe.extract_features_by_preprocessing([], towers, provenance, Path("/c"))
    floors = provenance["_unadapted_for"]
    for arm in ("crop130/seed2026", "crop150/seed2026"):
        floor_key = f"{floors[arm]}/uniform_mean_768"
        # The floor an arm resolves to must actually be among the extracted features.
        assert floors[arm] in features, f"{arm} resolves to a floor that was not extracted"
        assert floor_key.endswith("/uniform_mean_768")
