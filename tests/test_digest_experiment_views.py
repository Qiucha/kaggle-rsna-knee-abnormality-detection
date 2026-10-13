import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digest_site.pages import _experiment_views as views


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setattr(views.data, "load_curves", lambda root: None)
    arms = [
        {"id": key, "label": key, "group": "trained", "curve_root": "runs",
         "curve_run": key, "per_label": values, "baseline": "floor"}
        for key, values in (("reference", {"ACL": .7}), ("candidate", {"ACL": .8}), ("pending", {}))
    ]
    return arms, {"trained": {"label": "Trained", "color": "teal"}}, {
        "floor": {"label": "Unadapted", "per_label": {"ACL": .6}}}


def test_reference_deltas_use_each_evaluation_population(panel):
    arms, groups, baselines = panel
    _, blob, _ = views.experiment_views("test", arms, groups, labels=["ACL"], baselines=baselines,
        reference_id="reference", default_mode="reference",
        lm_test_per_label={"reference": {"ACL": .75}, "candidate": {"ACL": .9}})
    assert blob["defaultMode"] == "reference"
    assert blob["heatmaps"]["gold"]["reference"][0] == pytest.approx([None, 0, .1, None])
    assert blob["heatmaps"]["lm_test"]["reference"][0] == pytest.approx([None, 0, .15, None])
    assert blob["heatmaps"]["gold"]["level"][0] == [.6, .7, .8, None]
    assert "unadapted" not in blob["heatmaps"]["lm_test"]
    assert [m["key"] for m in blob["modes"]["lm_test"]] == ["reference", "level"]


def test_existing_gold_panels_keep_unadapted_default(panel):
    arms, groups, baselines = panel
    _, blob, _ = views.experiment_views("test", arms, groups, labels=["ACL"], baselines=baselines)
    assert blob["defaultMode"] == "unadapted"
    assert blob["heatmaps"]["gold"]["unadapted"][0] == pytest.approx([None, .1, .2, None])
    assert [choice["key"] for choice in blob["splitChoices"]] == ["gold"]
