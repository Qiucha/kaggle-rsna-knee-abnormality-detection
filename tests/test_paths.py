"""Where a machine's directories come from, and what happens when they are missing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contrastive_learning import cli, paths
from contrastive_learning.config import TrainConfig


def write_paths(directory: Path, **settings) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "paths.json"
    path.write_text(json.dumps(settings))
    return path


def test_explicit_beats_environment_beats_repository_default(monkeypatch, tmp_path):
    explicit = write_paths(tmp_path / "a", data_root="/explicit")
    environment = write_paths(tmp_path / "b", data_root="/environment")
    monkeypatch.setenv(paths.ENV_VAR, str(environment))
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "c")

    assert paths.load_settings(explicit)["data_root"] == Path("/explicit")
    assert paths.load_settings()["data_root"] == Path("/environment")

    monkeypatch.delenv(paths.ENV_VAR)
    assert paths.load_settings() == {}
    write_paths(tmp_path / "c", data_root="/repository")
    assert paths.load_settings()["data_root"] == Path("/repository")


def test_relative_values_resolve_against_the_configuration_not_the_cwd(monkeypatch, tmp_path):
    config = write_paths(tmp_path, data_root="mount", reports_csv="~/reports.csv")
    monkeypatch.chdir(tmp_path.parent)
    settings = paths.load_settings(config)
    assert settings["data_root"] == tmp_path / "mount"
    assert settings["reports_csv"] == Path.home() / "reports.csv"


def test_a_named_configuration_that_is_absent_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Paths configuration not found"):
        paths.load_settings(tmp_path / "absent.json")


def test_misspelled_settings_are_rejected_rather_than_ignored(tmp_path):
    config = write_paths(tmp_path, data_roots="/mount")
    with pytest.raises(ValueError, match="data_roots"):
        paths.load_settings(config)


def test_underscore_keys_are_comments(tmp_path):
    config = write_paths(tmp_path, _note="anything", data_root="/mount")
    assert paths.load_settings(config) == {"data_root": Path("/mount")}


def test_a_flag_overrides_the_configuration_which_overrides_the_default(tmp_path):
    config = write_paths(tmp_path, data_root="/mount", output_dir="/configured")
    base = ["--paths-config", str(config)]

    configured = cli.config_from_args(base)
    assert configured.data_root == Path("/mount")
    assert configured.output_dir == Path("/configured")
    assert configured.reports_csv == Path("data/train.csv")

    overridden = cli.config_from_args([*base, "--output-dir", "/flag"])
    assert overridden.output_dir == Path("/flag")



def test_the_configured_series_mirror_reaches_the_contrastive_config_as_a_path(tmp_path):
    config = write_paths(tmp_path, train_series_mirror="/mirror/train_series")
    configured = cli.config_from_args(["--paths-config", str(config)])
    assert configured.train_series_mirror == Path("/mirror/train_series")
    flagged = cli.config_from_args(["--paths-config", str(config), "--train-series-mirror", "/flag"])
    assert flagged.train_series_mirror == Path("/flag")

def test_an_unconfigured_mount_is_reported_where_it_is_needed(tmp_path):
    config = TrainConfig(cohort_manifest=tmp_path / "cohort.jsonl")
    # A manifest-driven run never resolves a mount, so it records neither derived path.
    assert "series_root" not in config.as_dict()
    with pytest.raises(ValueError, match=paths.DEFAULT_CONFIG_NAME):
        _ = config.series_root


def test_the_template_lists_exactly_the_supported_settings():
    template = json.loads(paths.template_path().read_text(encoding="utf-8"))
    assert {key for key in template if not key.startswith("_")} == set(paths.PATH_SETTINGS)


def test_repo_path_is_relative_inside_the_checkout_and_absolute_outside(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "checkpoints").mkdir(parents=True)
    elsewhere = tmp_path / "other-disk"
    elsewhere.mkdir()
    (root / "checkpoints" / "run").symlink_to(elsewhere)
    monkeypatch.setattr(paths, "repo_root", lambda: root)

    assert paths.repo_path(root / "checkpoints" / "run" / "best.pt") == "checkpoints/run/best.pt"
    assert paths.repo_path(elsewhere / "best.pt") == str(elsewhere / "best.pt")
