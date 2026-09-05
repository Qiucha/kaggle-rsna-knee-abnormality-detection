from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TEXT_MODEL_CHOICES = {
    "radbert-va": "UCSD-VA-health/RadBERT-RoBERTa-4m",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "radbert": "StanfordAIMI/RadBERT",
    "cxrbert": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "multilingual-e5": "intfloat/multilingual-e5-base",
}

MULTILINGUAL_TEXT_MODELS = {TEXT_MODEL_CHOICES["multilingual-e5"]}


@dataclass
class TrainConfig:
    """Serializable configuration for one training run."""

    data_root: Path = Path("/mnt/d")
    reports_csv: Path = Path("data/sample/train.csv")
    cohort_manifest: Path | None = None
    report_column: str = "Report"
    output_dir: Path = Path("checkpoints/contrastive")
    resume: Path | None = None
    text_model: str = TEXT_MODEL_CHOICES["radbert-va"]
    text_model_revision: str | None = None
    text_pooling: str = "cls"
    text_prefix: str = ""
    report_view: str = "full"
    image_model: str = "facebook/dinov2-small"
    image_model_revision: str | None = None
    image_size: int = 224
    crop_mm: float = 130.0
    slices_per_plane: int = 3
    slice_band_low: float = 0.20
    slice_band_high: float = 0.80
    max_text_length: int = 256
    embedding_dim: int = 256
    image_unfreeze_last: int = 3
    text_unfreeze_last: int = 2
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
    val_fraction: float = 0.10
    fold_count: int | None = None
    fold_index: int | None = None
    lr_image: float = 1e-5
    lr_text: float = 2e-5
    lr_projection: float = 1e-3
    weight_decay: float = 0.02
    warmup_fraction: float = 0.10
    grad_clip_norm: float = 1.0
    seed: int = 2026
    max_studies: int | None = None
    validate_pixels: bool = True
    cache_images: bool = False
    image_cache_dir: Path | None = None
    image_memory_cache_size: int = 0
    allow_text_policy_mismatch: bool = False
    trainable_checkpoints: bool = False
    local_files_only: bool = False
    amp: bool = True
    prepare_only: bool = False

    @property
    def series_csv(self) -> Path:
        return self.data_root / "train_series.csv"

    @property
    def series_root(self) -> Path:
        return self.data_root / "train_series"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        out["series_csv"] = str(self.series_csv)
        out["series_root"] = str(self.series_root)
        return out
