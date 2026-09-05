from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from contrastive_learning.config import TrainConfig
from contrastive_learning.selection import SELECTION_POLICY, verified_best_epoch


RUN_MANIFEST_VERSION = 1
_RESUME_MUTABLE_FIELDS = {"epochs", "resume", "output_dir", "prepare_only"}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    start_epoch: int
    best_score: float
    checkpoint: dict[str, Any] | None
    best_metrics: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for name in ("src", "scripts", "tests"):
        directory = root / name
        if directory.exists():
            paths.extend(path for path in directory.rglob("*.py") if path.is_file())
    config_directory = root / "configs"
    if config_directory.exists():
        paths.extend(path for path in config_directory.rglob("*") if path.is_file())
    paths.extend(path for path in root.glob("requirements*.txt") if path.is_file())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifacts(config: TrainConfig) -> dict[str, dict[str, str]]:
    candidates = {
        "reports_csv": config.reports_csv,
        "series_csv": config.series_csv,
        "cohort_manifest": config.cohort_manifest,
        "cohort_metadata": (
            config.cohort_manifest.with_suffix(".meta.json")
            if config.cohort_manifest is not None
            else None
        ),
    }
    return {
        name: {"path": str(path), "sha256": _file_hash(path)}
        for name, path in candidates.items()
        if path is not None and path.is_file()
    }


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=False, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "torch", "torchvision", "transformers", "pydicom"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    differences = {
        key: (saved.get(key), current.get(key))
        for key in saved.keys() | current.keys()
        if key not in _RESUME_MUTABLE_FIELDS and saved.get(key) != current.get(key)
    }
    if differences:
        detail = ", ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in differences.items())
        raise ValueError(f"Resume configuration is incompatible: {detail}")


def initialize_run(config: TrainConfig) -> RunContext:
    output_dir = Path(config.output_dir)
    config_value = config.as_dict()
    if config.resume is None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_dir.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"Run output already exists: {output_dir}; pass --resume with its last.pt"
            ) from error
        run_id = uuid.uuid4().hex
        _write_json_atomic(output_dir / "config.json", config_value)
        root = Path(__file__).resolve().parents[2]
        _write_json_atomic(
            output_dir / "run_manifest.json",
            {
                "schema_version": RUN_MANIFEST_VERSION,
                "selection_policy": SELECTION_POLICY,
                "run_id": run_id,
                "created_at": _now(),
                "target_epochs": config.epochs,
                "git": _git_metadata(root),
                "source_tree_sha256": _source_tree_hash(root),
                "source_artifacts": _source_artifacts(config),
                "dependencies": _dependency_versions(),
                "resume_events": [],
            },
        )
        return RunContext(run_id=run_id, start_epoch=1, best_score=-1.0, checkpoint=None)

    checkpoint_path = Path(config.resume).resolve()
    if checkpoint_path.parent != output_dir.resolve():
        raise ValueError("Resume checkpoint must be inside the configured output directory")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    saved_config_path = output_dir / "config.json"
    manifest_path = output_dir / "run_manifest.json"
    if not saved_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Resume requires config.json and run_manifest.json")
    saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
    _validate_resume_config(saved_config, config_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    completed_epoch = int(checkpoint["epoch"])
    history = _history(output_dir / "metrics.jsonl")
    if history and int(history[-1]["epoch"]) != completed_epoch:
        raise ValueError("metrics.jsonl does not end at the resume checkpoint epoch")
    if completed_epoch >= config.epochs:
        raise ValueError("Resume target epochs must exceed the checkpoint epoch")
    best_metrics = verified_best_epoch(output_dir / "metrics.jsonl", output_dir / "best.pt")
    best_score = float(best_metrics["mean_r@1"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("resume_events", []).append(
        {"at": _now(), "checkpoint": checkpoint_path.name, "target_epochs": config.epochs}
    )
    manifest["target_epochs"] = config.epochs
    _write_json_atomic(manifest_path, manifest)
    return RunContext(
        run_id=manifest["run_id"],
        start_epoch=completed_epoch + 1,
        best_score=best_score,
        checkpoint=checkpoint,
        best_metrics=best_metrics,
    )


def update_run_manifest(output_dir: Path, values: dict[str, Any]) -> None:
    path = Path(output_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(values)
    _write_json_atomic(path, manifest)
