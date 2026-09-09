"""One serializable description of a training run.

:class:`TrainConfig` is the single object that decides what a run does: which
studies, which text view, which backbones, which optimizer settings, which
checkpoint format. It is written verbatim into every run's ``config.json`` and into
every checkpoint, and a resume compares the two field by field. Two consequences
follow, and both are load-bearing:

* **Fields are append-only in practice.** Removing or renaming one makes a resume
  of any existing run fail, because the saved and current field sets would differ.
* **Derived values belong in properties**, not in fields, so that reading an old
  ``config.json`` cannot disagree with the code about how a value is computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from contrastive_learning import paths
from contrastive_learning.text import REPORT_VIEWS


# Short names accepted on the command line for the text encoders under comparison.
# The values are the exact Hugging Face identifiers; revisions are pinned
# separately, per run, because a name alone does not identify weights.
TEXT_MODEL_CHOICES = {
    "radbert-va": "UCSD-VA-health/RadBERT-RoBERTa-4m",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "radbert": "StanfordAIMI/RadBERT",
    "cxrbert": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "multilingual-e5": "intfloat/multilingual-e5-base",
}

# Encoders that can read the cohort's reports in their original languages. A
# ``multilingual_raw`` cohort trained with anything else is a silent error, so
# load_cohort refuses the combination unless it is overridden explicitly.
MULTILINGUAL_TEXT_MODELS = {TEXT_MODEL_CHOICES["multilingual-e5"]}

# Fields holding filesystem paths, restored to Path objects when reading a saved
# configuration back from JSON.
_PATH_FIELDS = (
    "data_root",
    "reports_csv",
    "cohort_manifest",
    "output_dir",
    "image_cache_dir",
    "resume",
)

DEFAULT_CACHE_STUDIES = 64


@dataclass
class TrainConfig:
    """Serializable configuration for one training run."""

    # --- Data -------------------------------------------------------------- #
    # No default: the DICOM mount is a different directory on every machine, so it
    # comes from --data-root or the paths configuration (see ``paths.py``). It stays
    # None whenever a run reads a frozen cohort manifest, which needs no mount.
    data_root: Path | None = None
    reports_csv: Path = Path("data/sample/train.csv")
    # Preferred input: a frozen cohort with fixed patient-grouped splits. When it
    # is None the legacy CSV path pairs reports and series and splits on the fly.
    cohort_manifest: Path | None = None
    report_column: str = "Report"
    output_dir: Path = Path("checkpoints/contrastive")
    resume: Path | None = None

    # --- Text tower -------------------------------------------------------- #
    text_model: str = TEXT_MODEL_CHOICES["radbert-va"]
    text_model_revision: str | None = None
    text_pooling: str = "cls"
    # Prepended to every report; some encoders (E5) expect such an instruction.
    text_prefix: str = ""
    report_view: str = "full"
    max_text_length: int = 256

    # --- Image tower and preprocessing ------------------------------------- #
    image_model: str = "facebook/dinov2-small"
    image_model_revision: str | None = None
    image_size: int = 224
    crop_mm: float = 130.0
    slices_per_plane: int = 3
    slice_band_low: float = 0.20
    slice_band_high: float = 0.80

    # --- Optimization ------------------------------------------------------ #
    embedding_dim: int = 256
    # Backbones are frozen except for their last N transformer blocks: the cohort
    # is far too small to fine-tune either encoder end to end.
    image_unfreeze_last: int = 3
    text_unfreeze_last: int = 2
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
    lr_image: float = 1e-5
    lr_text: float = 2e-5
    lr_projection: float = 1e-3
    weight_decay: float = 0.02
    warmup_fraction: float = 0.10
    grad_clip_norm: float = 1.0
    seed: int = 2026

    # --- Splitting (legacy CSV path only) ---------------------------------- #
    val_fraction: float = 0.10
    fold_count: int | None = None
    fold_index: int | None = None

    # --- Execution --------------------------------------------------------- #
    max_studies: int | None = None
    validate_pixels: bool = True
    # cache_images is the older switch meaning "cache DEFAULT_CACHE_STUDIES per
    # worker"; image_memory_cache_size supersedes it with an explicit count. Both
    # are retained so saved configurations stay readable; memory_cache_size
    # resolves them.
    cache_images: bool = False
    image_cache_dir: Path | None = None
    image_memory_cache_size: int = 0
    allow_text_policy_mismatch: bool = False
    trainable_checkpoints: bool = False
    local_files_only: bool = False
    amp: bool = True
    # Index and summarize the cohort, then stop before building any model.
    prepare_only: bool = False

    def validate_training(self) -> None:
        """Reject unsupported settings before indexing data or creating a run.

        These run first, so a misconfiguration costs seconds rather than failing
        after a cohort scan or partway through the first epoch.
        """
        if self.slices_per_plane != 3:
            raise ValueError("DINOv2 requires exactly three sampled slices as input channels")
        if self.image_size % 14:
            raise ValueError("DINOv2 image_size must be divisible by its 14-pixel patch size")
        if self.batch_size < 2:
            # In-batch negatives are the training signal; a batch of one has none.
            raise ValueError("Contrastive training requires batch_size >= 2")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.image_unfreeze_last < 0 or self.text_unfreeze_last < 0:
            raise ValueError("unfreeze counts cannot be negative")
        if (self.fold_count is None) != (self.fold_index is None):
            raise ValueError("fold_count and fold_index must be provided together")
        if self.max_studies is not None and self.max_studies < 1:
            raise ValueError("max_studies must be positive")
        if self.image_memory_cache_size < 0:
            raise ValueError("image_memory_cache_size cannot be negative")
        if self.text_pooling not in {"cls", "mean"}:
            raise ValueError("text_pooling must be 'cls' or 'mean'")
        if self.report_view not in REPORT_VIEWS:
            raise ValueError(f"report_view must be one of {REPORT_VIEWS}")
        if self.trainable_checkpoints and (
            self.image_model_revision is None or self.text_model_revision is None
        ):
            # A compact checkpoint stores only trained weights, so the frozen ones
            # must be recoverable from exactly pinned pretrained backbones.
            raise ValueError("trainable checkpoints require pinned image and text model revisions")

    def require_data_root(self) -> Path:
        """The DICOM mount, or an error naming every way to configure one.

        Raised at the moment a mount is actually needed rather than at construction,
        because a run driven by a frozen cohort manifest never touches one.
        """
        if self.data_root is None:
            raise ValueError(paths.missing_data_root_message())
        return Path(self.data_root)

    @property
    def series_csv(self) -> Path:
        """Series inventory of the DICOM mount (legacy CSV path)."""
        return paths.series_csv(self.require_data_root())

    @property
    def series_root(self) -> Path:
        """Root of the ``<study>/<series>/*.dcm`` tree on the DICOM mount."""
        return paths.series_root(self.require_data_root())

    @property
    def image_kwargs(self) -> dict[str, Any]:
        """Image preprocessing settings shared by training, evaluation, and caching."""
        return {
            "image_size": self.image_size,
            "crop_mm": self.crop_mm,
            "slices_per_plane": self.slices_per_plane,
            "slice_band": (self.slice_band_low, self.slice_band_high),
        }

    @property
    def memory_cache_size(self) -> int:
        """Study tensors each loader worker may retain, resolving both cache flags."""
        return self.image_memory_cache_size or (DEFAULT_CACHE_STUDIES if self.cache_images else 0)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainConfig:
        """Read saved settings, ignoring derived paths and any extra metadata.

        Tolerating unknown keys is what lets a checkpoint written by a later version
        still be evaluated, and lets ``as_dict`` include derived values for readers.
        """
        names = {field.name for field in fields(cls)}
        values = {key: item for key, item in value.items() if key in names}
        for name in _PATH_FIELDS:
            if values.get(name) is not None:
                values[name] = Path(values[name])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready settings, plus the derived data paths, for run artifacts."""
        out = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
        # Only derivable with a mount; a manifest-driven run records neither, which
        # is also what makes such a run resumable on a host without the mount.
        if self.data_root is not None:
            out["series_csv"] = str(self.series_csv)
            out["series_root"] = str(self.series_root)
        return out
