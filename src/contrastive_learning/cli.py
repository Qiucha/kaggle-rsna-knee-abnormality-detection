"""Command-line surface for one training run.

This module does three things and nothing else: describe every setting as a flag,
translate the parsed values into a :class:`TrainConfig`, and check that the data it
names exists before handing control to :func:`run_training`. All policy about what
those settings mean lives in the modules that own them -- keeping this file thin is
what allows an experiment matrix to drive training purely through flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from contrastive_learning import paths
from contrastive_learning.config import TEXT_MODEL_CHOICES, TrainConfig
from contrastive_learning.text import REPORT_VIEWS
from contrastive_learning.training import run_training


# Fallbacks for path settings stated neither on the command line nor in a paths
# configuration, read from the dataclass so the two cannot drift apart. They are
# repository-relative, hence portable as they stand; the settings missing from this
# mapping -- data_root above all -- have no sensible default and must be configured.
_TRAIN_FIELDS = {field.name for field in fields(TrainConfig)}
_PATH_DEFAULTS = {
    field.name: field.default
    for field in fields(TrainConfig)
    if field.name in paths.PATH_SETTINGS and field.default is not None
}


def _path_help(name: str, description: str) -> str:
    """Describe a path flag and the whole chain behind it.

    These flags default to ``SUPPRESS`` rather than to a value, so that an unstated
    setting can fall through to the paths configuration. That also stops
    ``ArgumentDefaultsHelpFormatter`` from advertising a misleading "default: None",
    which is why each one has to state its own fallbacks here.
    """
    default = _PATH_DEFAULTS.get(name)
    fallback = f"then {default}" if default is not None else "and has no default"
    return f"{description}. Unset, falls back to '{name}' in the paths configuration, {fallback}"


def build_parser() -> argparse.ArgumentParser:
    """Define every training flag. Defaults mirror :class:`TrainConfig`."""
    parser = argparse.ArgumentParser(
        description="Train a model to match knee MRI studies with their reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--paths-config",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "JSON file assigning this machine's directories; defaults to "
            f"{paths.DEFAULT_CONFIG_NAME} at the repository root or ${paths.ENV_VAR}. "
            f"See {paths.TEMPLATE_RELATIVE_PATH}"
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        type=Path,
        default=argparse.SUPPRESS,
        help=_path_help("data_root", "Root of the DICOM mount"),
    )
    data.add_argument(
        "--reports-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help=_path_help("reports_csv", "CSV of study reports"),
    )
    data.add_argument(
        "--cohort-manifest",
        type=Path,
        default=argparse.SUPPRESS,
        help=_path_help(
            "cohort_manifest", "JSONL of fixed train/validation/test assignments"
        ),
    )
    data.add_argument("--report-column", default="Report")
    data.add_argument("--max-studies", type=int)
    data.add_argument(
        "--skip-decode-validation",
        action="store_true",
        help="Trust that every indexed study decodes instead of reading its pixels now",
    )

    labels = parser.add_argument_group("label-similarity targets")
    labels.add_argument(
        "--lm-label-csv",
        type=Path,
        help="LM-extracted finding labels used to soften the contrastive target",
    )
    labels.add_argument(
        "--label-similarity-weight",
        type=float,
        default=0.0,
        help="Fraction of the target taken from label agreement; 0.0 is plain CLIP-style training",
    )

    labels.add_argument(
        "--label-findings",
        nargs="+",
        metavar="FINDING",
        help="Restrict the label target to these findings; default uses every finding in the CSV",
    )

    text = parser.add_argument_group("text tower")
    text.add_argument(
        "--text-model",
        default="radbert-va",
        help="A short name (%s) or Hugging Face model ID" % ", ".join(TEXT_MODEL_CHOICES),
    )
    text.add_argument("--text-model-revision")
    text.add_argument("--text-pooling", choices=("cls", "mean"), default="cls")
    text.add_argument("--text-prefix", default="", help="Prepended to every report, e.g. 'query: '")
    text.add_argument("--report-view", choices=REPORT_VIEWS, default="full")
    text.add_argument("--max-text-length", type=int, default=256)
    text.add_argument("--text-unfreeze-last", type=int, default=2)
    text.add_argument(
        "--allow-text-policy-mismatch",
        action="store_true",
        help="Permit a non-multilingual encoder on a multilingual_raw cohort",
    )

    image = parser.add_argument_group("image tower")
    image.add_argument("--image-model", default="facebook/dinov2-small")
    image.add_argument("--image-model-revision")
    image.add_argument("--image-unfreeze-last", type=int, default=3)
    image.add_argument(
        "--image-plane-pooling", choices=("attention", "uniform"), default="attention"
    )
    image.add_argument("--image-drop-path-rate", type=float, default=0.0)
    # Physical extent of the centre crop. read_slot crops or pads to exactly this,
    # so every series lands at crop_mm / image_size millimetres per pixel.
    image.add_argument("--crop-mm", type=float, default=130.0)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--embedding-dim", type=int, default=256)
    optimization.add_argument("--epochs", type=int, default=20)
    optimization.add_argument("--batch-size", type=int, default=16)
    optimization.add_argument("--lr-image", type=float, default=1e-5)
    optimization.add_argument("--lr-text", type=float, default=2e-5)
    optimization.add_argument("--lr-projection", type=float, default=1e-3)
    optimization.add_argument(
        "--max-inverse-temperature",
        type=float,
        default=100.0,
        help="Upper clamp on learnable inverse temperature; default 100.0",
    )
    optimization.add_argument(
        "--freeze-temperature",
        action="store_true",
        help="Keep inverse temperature frozen at initial value (~14.28, tau=0.07)",
    )
    optimization.add_argument("--seed", type=int, default=2026)
    optimization.add_argument("--no-amp", action="store_true")

    lora = parser.add_argument_group("lora adaptation")
    lora.add_argument("--use-lora", action="store_true", help="Apply LoRA adapters to backbones")
    lora.add_argument("--lora-rank", type=int, default=16, help="LoRA rank dimension")
    lora.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling factor alpha")
    lora.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability")

    memory = parser.add_argument_group("memory optimization")
    memory.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on backbones to trade compute for memory",
    )
    memory.add_argument(
        "--save-on-cpu",
        action="store_true",
        help="Offload backward activations to host RAM using torch.autograd.graph.save_on_cpu",
    )

    declip = parser.add_argument_group("declip based alignment")
    declip.add_argument(
        "--use-declip",
        action="store_true",
        help="Enable DeCLIP multi-view, nearest-neighbour report, and unimodal supervision",
    )
    declip.add_argument(
        "--declip-mv-weight",
        type=float,
        default=1.0,
        help="Weight for multi-view cross-modal InfoNCE loss L_MV",
    )
    declip.add_argument(
        "--declip-nn-weight",
        type=float,
        default=0.5,
        help="Weight for nearest-neighbour report contrastive loss L_NN",
    )
    declip.add_argument(
        "--declip-image-ss-weight",
        type=float,
        default=0.5,
        help="Weight for image self-supervision (SimCLR) loss L_I",
    )
    declip.add_argument(
        "--declip-queue-size",
        type=int,
        default=512,
        help="Capacity of the FIFO queue for nearest-neighbour report embeddings",
    )
    declip.add_argument(
        "--declip-text-mask-prob",
        type=float,
        default=0.15,
        help="Token masking probability for report augmentation R'",
    )

    splitting = parser.add_argument_group("splitting (ignored with --cohort-manifest)")
    splitting.add_argument("--val-fraction", type=float, default=0.10)
    splitting.add_argument(
        "--fold-count", type=int, help="Use deterministic K-fold validation instead of --val-fraction"
    )
    splitting.add_argument(
        "--fold-index", type=int, help="Zero-based validation fold used with --fold-count"
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=_path_help("output_dir", "Where this run writes checkpoints and metrics"),
    )
    execution.add_argument("--num-workers", type=int, default=4)
    execution.add_argument(
        "--cache-images", action="store_true", help="Keep up to 64 preprocessed studies in memory"
    )
    execution.add_argument(
        "--image-memory-cache-size",
        type=int,
        default=0,
        help="Maximum preprocessed study tensors held in each loader process",
    )
    execution.add_argument(
        "--image-cache-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=_path_help(
            "image_cache_dir", "Save preprocessed images on disk for reuse across runs"
        ),
    )
    execution.add_argument("--resume", type=Path, help="Resume from last.pt in --output-dir")
    execution.add_argument(
        "--trainable-checkpoints",
        action="store_true",
        help="Save only trained weights; requires exact image and text model revisions",
    )
    execution.add_argument("--local-files-only", action="store_true")
    execution.add_argument(
        "--prepare-only", action="store_true", help="Index and summarize the cohort, then stop"
    )
    return parser


def _apply_path_settings(values: dict[str, Any]) -> None:
    """Fill in unstated path settings from the paths configuration, then the defaults.

    Resolution happens here, before a :class:`TrainConfig` exists, so that everything
    downstream -- and every artifact written from the configuration -- sees concrete
    paths and never has to know a file supplied them. A flag always wins: that is what
    lets a frozen experiment matrix state its own output directory while still picking
    up this machine's DICOM mount.
    """
    configured = paths.load_settings(values.pop("paths_config", None))
    for name in paths.PATH_SETTINGS:
        # Settings for other entry points (the fine-tuning mirror) are not TrainConfig fields.
        if name not in _TRAIN_FIELDS:
            continue
        if values.get(name) is None:
            values[name] = configured.get(name, _PATH_DEFAULTS.get(name))


def config_from_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse arguments into a validated configuration.

    Path settings are layered over the paths configuration, two flags are stated
    negatively for a safe default but stored positively, and the memory cache size is
    resolved here so ``config.json`` records the number that was actually used rather
    than the flags it was derived from.
    """
    values = vars(build_parser().parse_args(argv))
    _apply_path_settings(values)
    values["text_model"] = TEXT_MODEL_CHOICES.get(values["text_model"], values["text_model"])
    values["validate_pixels"] = not values.pop("skip_decode_validation")
    values["amp"] = not values.pop("no_amp")
    config = TrainConfig(**values)
    config.image_memory_cache_size = config.memory_cache_size
    return config


def _missing_data_paths(config: TrainConfig) -> list[Path]:
    """Inputs that must exist: a cohort and its metadata, or the legacy CSVs.

    The legacy CSV path names the DICOM mount, so its absence is reported here as a
    configuration error rather than surfacing later as a missing directory.
    """
    required = (
        (config.cohort_manifest, config.cohort_manifest.with_suffix(".meta.json"))
        if config.cohort_manifest is not None
        else (config.reports_csv, config.series_csv, config.series_root)
    )
    return [path for path in required if path is not None and not path.exists()]


def main() -> None:
    try:
        config = config_from_args()
        missing = _missing_data_paths(config)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if missing:
        raise SystemExit("Missing required data path(s): " + ", ".join(map(str, missing)))
    print(json.dumps(run_training(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
