"""Where this machine keeps its data, kept out of the code.

Every other module names paths *relatively* -- ``data/train.csv``,
``checkpoints/contrastive`` -- except one: the DICOM mount, which is a different
directory on every machine that has ever run this project. Hard-coding it meant
that a checkout only worked on the host it was written on, and that a frozen
experiment matrix silently depended on a default it never stated.

This module owns that resolution and nothing else. It reads one small JSON file,
the *paths configuration*, whose keys are exactly the settings a machine is
allowed to relocate. Three properties are load-bearing:

* **It is read once, explicitly, by a command-line entry point** -- never at import
  time, and never deep inside a workflow. Everything downstream still receives a
  fully resolved :class:`~contrastive_learning.config.TrainConfig`, so run
  artifacts keep recording concrete paths and remain reproducible. The one
  exception is :func:`host_series_root`: cohort manifests name series only as
  ``<study>/<series>``, so reading one needs this host's tree, and a checkpoint's
  recorded mount may no longer exist (docs/adr/0001).
* **A flag always wins over the file.** The file supplies defaults for settings the
  caller did not state, which is what lets an existing matrix run unchanged on a
  new host without editing the frozen commands inside it.
* **Relative values resolve against the configuration file's own directory**, not
  the working directory, so a paths file describes the same locations no matter
  where a command is invoked from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# Environment variable naming a paths configuration, for hosts that keep it
# outside the checkout (a shared cluster image, a container mount).
ENV_VAR = "CONTRASTIVE_PATHS_CONFIG"

# Looked for at the repository root when nothing else names a file. It is
# git-ignored: a paths file describes one machine and must never be committed.
DEFAULT_CONFIG_NAME = "paths.json"

# Committed example, quoted in the error raised when a required path is missing.
TEMPLATE_RELATIVE_PATH = Path("configs/paths.template.json")

# The settings a machine may relocate. Deliberately short: everything absent from
# this tuple is a scientific choice that belongs in an experiment configuration,
# not a property of the host.
PATH_SETTINGS = (
    "data_root",
    "reports_csv",
    "cohort_manifest",
    "output_dir",
    "image_cache_dir",
)


def repo_root() -> Path:
    """Root of the checkout, derived from this file rather than the caller's cwd."""
    return Path(__file__).resolve().parents[2]


def template_path() -> Path:
    return repo_root() / TEMPLATE_RELATIVE_PATH


def series_csv(data_root: Path) -> Path:
    """Series inventory of the DICOM mount."""
    return Path(data_root) / "train_series.csv"


def series_root(data_root: Path) -> Path:
    """Root of the ``<study>/<series>/*.dcm`` tree on the DICOM mount."""
    return Path(data_root) / "train_series"


def host_data_root(explicit: Path | None = None) -> Path:
    """This machine's DICOM mount, from the paths configuration."""
    data_root = load_settings(explicit).get("data_root")
    if data_root is None:
        raise ValueError(missing_data_root_message())
    return data_root


def host_series_root(explicit: Path | None = None) -> Path:
    """This machine's ``train_series`` tree, under which cohort manifest slots resolve."""
    return series_root(host_data_root(explicit))


def find_config(explicit: Path | None = None) -> Path | None:
    """First paths configuration that applies, or None when the host states none.

    Order is most specific first: an explicit argument, then the environment, then
    ``paths.json`` beside the checkout. An explicit path or an environment value
    that does not exist is an error rather than a silent fallback -- the caller
    asked for that file specifically.
    """
    for candidate in (explicit, os.environ.get(ENV_VAR)):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Paths configuration not found: {path}")
            return path
    default = repo_root() / DEFAULT_CONFIG_NAME
    return default if default.is_file() else None


def load_settings(explicit: Path | None = None) -> dict[str, Path]:
    """Read the applicable paths configuration into resolved :class:`Path` values.

    Unknown keys are rejected rather than ignored: a typo in a path name would
    otherwise leave a run reading the built-in default and only reveal itself as a
    confusing "file not found" much later. Keys beginning with ``_`` are free for
    comments, which JSON has no syntax for.
    """
    path = find_config(explicit)
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Paths configuration must be a JSON object: {path}")
    settings = {key: item for key, item in value.items() if not key.startswith("_")}
    if unknown := sorted(set(settings) - set(PATH_SETTINGS)):
        raise ValueError(
            f"Unknown path setting(s) in {path}: {', '.join(unknown)}; "
            f"supported keys are {', '.join(PATH_SETTINGS)}"
        )
    return {
        key: _resolve(item, relative_to=path.parent)
        for key, item in settings.items()
        if item is not None
    }


def _resolve(value: str, *, relative_to: Path) -> Path:
    """Expand ``~`` and anchor a relative value to the configuration's directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (relative_to / path)


def missing_data_root_message() -> str:
    """One explanation of how to supply the DICOM mount, used wherever it is needed."""
    return (
        "No data_root is configured, but this operation reads the DICOM mount. "
        f"Create {DEFAULT_CONFIG_NAME} at the repository root (copy "
        f"{TEMPLATE_RELATIVE_PATH}), set {ENV_VAR} to a paths configuration, or name "
        "the mount on the command line (--data-root, or --series-csv/--series-root)."
    )
