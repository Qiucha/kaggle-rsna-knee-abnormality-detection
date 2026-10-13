"""Own the checkpoint file format, epoch selection, and run provenance.

Callers of this module never name a key inside a checkpoint. They hand over
their live objects to :func:`save_checkpoint`, and get them restored by
:func:`load_model_weights` and :func:`restore_training_state`. That keeps the
payload layout -- including the two supported weight formats and the four random
number generator states that make a resume bit-reproducible -- changeable here
alone.

Two weight formats exist:

``full_v1``
    The entire ``state_dict``. Self-contained: reloadable without network access
    to the pretrained backbones.
``trainable_delta_v1``
    Only parameters that required gradients. Roughly an order of magnitude
    smaller, at the cost of needing the exact pinned backbone revisions at
    reload time, which is why :meth:`TrainConfig.validate_training` demands them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from contrastive_learning import paths
from contrastive_learning.artifacts import (
    read_json,
    read_jsonl,
    sha256_file,
    write_atomic,
    write_json,
)
from contrastive_learning.config import TrainConfig


RUN_MANIFEST_VERSION = 1

# The one authoritative statement of how the best epoch is chosen. Frozen
# experiment matrices copy this string and refuse to run when it changes, so a
# selection rule can never be altered underneath a comparison already in flight.
SELECTION_POLICY = "mean_r@1 descending, validation_loss ascending, earliest epoch"

# Settings a resume is allowed to differ on: the new epoch target, the pointer to
# the checkpoint, the directory being continued, the indexing-only shortcut, and
# where this host keeps the DICOM tree and its optional byte-identical local
# mirror (series are matched by <study>/<series>, so moving the mount or adding
# a mirror does not change the data). Everything else must match, or the
# continued run would not be the same run.
_RESUME_MUTABLE_FIELDS = frozenset({
    "epochs", "resume", "output_dir", "prepare_only",
    "data_root", "series_csv", "series_root", "train_series_mirror",
})


# --------------------------------------------------------------------------- #
# Epoch selection
# --------------------------------------------------------------------------- #


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, int]:
    """Rank one epoch's metrics under SELECTION_POLICY; larger tuples are better.

    Loss and epoch are negated so a single ``max``/``>`` comparison expresses
    "highest recall, then lowest loss, then earliest epoch".
    """
    recall = float(metrics["mean_r@1"])
    loss = float(metrics["validation_loss"])
    if not math.isfinite(recall) or not math.isfinite(loss):
        raise ValueError("Checkpoint selection requires finite recall and validation loss")
    return recall, -loss, -int(metrics["epoch"])


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Whether ``candidate`` should replace ``incumbent`` as the run's best epoch.

    The candidate key is computed before the ``incumbent is None`` shortcut so the
    very first epoch of a run still fails loudly on non-finite metrics rather than
    being adopted as the best.
    """
    key = selection_key(candidate)
    return incumbent is None or key > selection_key(incumbent)


def verified_best_epoch(history_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Re-derive the best epoch from history and prove ``best.pt`` is that epoch.

    Recomputing from ``metrics.jsonl`` rather than trusting the saved file is what
    catches a checkpoint written under a superseded selection rule, or a history
    that was truncated or appended to out of band.
    """
    rows = read_jsonl(history_path)
    if not rows:
        raise ValueError("Checkpoint selection requires nonempty metrics history")
    best = max(rows, key=selection_key)
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["epoch"] != best["epoch"] or checkpoint["metrics"] != best:
        raise ValueError("best.pt does not match the selected validation epoch and metrics")
    return best


# --------------------------------------------------------------------------- #
# Checkpoint reading and writing
# --------------------------------------------------------------------------- #


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint as an opaque handle, without executing pickled code from it."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def checkpoint_config(checkpoint: dict[str, Any]) -> TrainConfig:
    """Recover the settings a checkpoint was trained under."""
    return TrainConfig.from_dict(checkpoint["config"])


def checkpoint_epoch(checkpoint: dict[str, Any]) -> int:
    """Recover the epoch a checkpoint completed."""
    return int(checkpoint["epoch"])


def checkpoint_metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Recover the metrics stored beside a checkpoint's weights.

    A checkpoint written before metrics were recorded carries none, so the
    absent case is a normal read rather than a corrupt payload.
    """
    return checkpoint.get("metrics", {})


def has_full_weights(checkpoint: dict[str, Any]) -> bool:
    """Whether the payload carries every weight, or only the trained subset.

    Evaluation uses this to decide whether pretrained backbones must be
    downloaded to fill in the frozen parameters a compact checkpoint omits.
    """
    return "model_state_dict" in checkpoint


def load_model_weights(model: torch.nn.Module, checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Restore weights from either supported format.

    A compact checkpoint is loaded non-strictly on purpose: the missing keys are
    exactly the frozen parameters the freshly built backbones already hold.
    """
    if has_full_weights(checkpoint):
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    return model


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
) -> None:
    """Restore everything needed for the next epoch to continue the same run.

    Beyond weights and optimizer/scheduler/scaler state this includes the loader's
    shuffle generator and the global CPU and CUDA generators, so augmentation and
    batch order after a resume match an uninterrupted run.
    """
    load_model_weights(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("loader_generator_state") is not None:
        loader_generator.set_state(checkpoint["loader_generator_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    config: TrainConfig,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    best_score: float,
    run_id: str,
) -> None:
    """Write one resumable checkpoint atomically, in the format ``config`` requests."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "metrics": metrics,
        "config": config.as_dict(),
        "run_id": run_id,
        "best_score": best_score,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if config.trainable_checkpoints:
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        payload["state_dict_format"] = "trainable_delta_v1"
        payload["trainable_state_dict"] = {
            name: value for name, value in model.state_dict().items() if name in trainable
        }
    else:
        payload["state_dict_format"] = "full_v1"
        payload["model_state_dict"] = model.state_dict()
    write_atomic(path, lambda temporary: torch.save(payload, temporary))


# --------------------------------------------------------------------------- #
# Run provenance and resume
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunContext:
    """Where a run resumes from, and what it has already achieved."""

    run_id: str
    start_epoch: int
    best_score: float
    checkpoint: dict[str, Any] | None
    best_metrics: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_tree_hash(root: Path) -> str:
    """Hash the code and configuration that define a run's behaviour.

    Git revision alone is insufficient here because runs are routinely launched
    from an uncommitted working tree; this digest identifies the exact bytes.
    """
    digest = hashlib.sha256()
    files: list[Path] = []
    for name in ("src", "scripts", "tests"):
        directory = root / name
        if directory.exists():
            files.extend(path for path in directory.rglob("*.py") if path.is_file())
    if (root / "configs").exists():
        files.extend(path for path in (root / "configs").rglob("*") if path.is_file())
    files.extend(path for path in root.glob("requirements*.txt") if path.is_file())
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_artifacts(config: TrainConfig) -> dict[str, dict[str, str]]:
    """Hash the data inputs, so a silently edited cohort is detectable later."""
    candidates = {
        "reports_csv": config.reports_csv,
        # Absent for a manifest-driven run, which never resolves a DICOM mount.
        "series_csv": config.series_csv if config.data_root is not None else None,
        "cohort_manifest": config.cohort_manifest,
        "cohort_metadata": (
            config.cohort_manifest.with_suffix(".meta.json")
            if config.cohort_manifest is not None
            else None
        ),
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
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


def _validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    # A setting the saved config lacks was added to TrainConfig after that run
    # started, so the run trained with its default; only a non-default value now
    # is a real change.
    defaults = TrainConfig().as_dict()
    saved = {**{key: defaults[key] for key in defaults.keys() - saved.keys()}, **saved}
    differences = {
        key: (saved.get(key), current.get(key))
        for key in saved.keys() | current.keys()
        if key not in _RESUME_MUTABLE_FIELDS and saved.get(key) != current.get(key)
    }
    if differences:
        detail = ", ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in differences.items())
        raise ValueError(f"Resume configuration is incompatible: {detail}")


def initialize_run(config: TrainConfig) -> RunContext:
    """Create a fresh run directory, or validate and continue an existing one.

    A fresh run refuses to write into an existing directory: overwriting one is
    almost always an accident, and recovering the intended behaviour is spelled
    out in the error. A resume must point at a checkpoint inside its own output
    directory, agree with the saved configuration on every immutable field, end
    at the last recorded epoch, and have a ``best.pt`` that still wins under the
    current selection policy -- checked before any training work begins.
    """
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
        write_json(output_dir / "config.json", config_value)
        root = paths.repo_root()
        write_json(
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
    _validate_resume_config(read_json(saved_config_path), config_value)
    checkpoint = read_checkpoint(checkpoint_path)
    completed_epoch = int(checkpoint["epoch"])
    history_path = output_dir / "metrics.jsonl"
    history = read_jsonl(history_path) if history_path.exists() else []
    if history and int(history[-1]["epoch"]) != completed_epoch:
        raise ValueError("metrics.jsonl does not end at the resume checkpoint epoch")
    if completed_epoch >= config.epochs:
        raise ValueError("Resume target epochs must exceed the checkpoint epoch")
    best_metrics = verified_best_epoch(history_path, output_dir / "best.pt")
    manifest = read_json(manifest_path)
    manifest.setdefault("resume_events", []).append(
        {"at": _now(), "checkpoint": checkpoint_path.name, "target_epochs": config.epochs}
    )
    manifest["target_epochs"] = config.epochs
    write_json(manifest_path, manifest)
    return RunContext(
        run_id=manifest["run_id"],
        start_epoch=completed_epoch + 1,
        best_score=float(best_metrics["mean_r@1"]),
        checkpoint=checkpoint,
        best_metrics=best_metrics,
    )


def update_run_manifest(output_dir: Path, values: dict[str, Any]) -> None:
    """Merge resolved facts (such as backbone revisions) into an existing manifest."""
    path = Path(output_dir) / "run_manifest.json"
    manifest = read_json(path)
    manifest.update(values)
    write_json(path, manifest)
