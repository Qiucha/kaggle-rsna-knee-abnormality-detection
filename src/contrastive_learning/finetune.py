"""End-to-end supervised fine-tuning of the image backbone on LM labels (ladder R1-R3).

R0 fits heads on a frozen contrastive encoder. This module instead trains the last
blocks of a pretrained image backbone together with a dense head, using masked BCE
against the LM labels. No text tower is involved.

Three things keep a rung comparable with R0 and with the other rungs:

* **Same populations.** Train, validation and gold come from
  :func:`lm_supervision.load_population` on R0's gold-holdout manifest. The LM-test
  population comes from :func:`lm_proxy.load_test_population`. So every rung is
  scored on the same studies as R0, and gold never enters training or selection.
* **Same selection rule.** The epoch is chosen on validation masked BCE against
  the hard verdicts, whatever target the rung trains on. A soft-label rung therefore
  cannot win by redefining its own yardstick.
* **Same deployment path.** The forward pass is :func:`submission.image_features`
  with gradients, and the head is exported as ``dense_v1`` with an identity scaler.
  The saved member therefore loads and predicts through the existing Kaggle code.

UNK verdicts are masked unless ``unk_weight`` is set (see :func:`unk_supervision`).
``targets: "soft"`` swaps only the observed cells' 0/1 for the CSV's ``<finding>`` value. That value is a lookup from verdict and
confidence (NO is always 0.08, YES is 0.68-0.94), not a calibrated probability.
"""

from __future__ import annotations

import json
import math
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import paths
from .artifacts import read_json, sha256_file, sha256_text, write_atomic, write_json
from .data import KneeReportDataset, slot_identity
from .lm_proxy import load_test_population, masked_macro_auroc
from .lm_supervision import load_population
from .preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    apply_left_knee,
    apply_left_knee_windows,
    augment_study_images,
    read_slot,
    read_windows,
)
from .model import _IMAGENET_MEAN, _IMAGENET_STD, freeze_then_unfreeze_last
from .submission import LABELS, UID, export_member, pool_windows, score_predictions
from .supervised_head import masked_bce

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("verdict", "soft")
SLICE_SAMPLING = ("fixed", "random")
POOLING = ("mean", "attention", "grouped")
IMAGE_AUGMENTATION_DEFAULTS = {
    "rotation_degrees": 8.0, "zoom": 0.08, "shift": 0.05, "intensity": 0.10,
}


def image_augmentation_settings(config: dict) -> dict | None:
    """An optional augmentation object; omission disables it, an empty object uses defaults."""
    if "image_augmentation" not in config:
        return None
    spec = config["image_augmentation"]
    if not isinstance(spec, dict) or set(spec) - set(IMAGE_AUGMENTATION_DEFAULTS):
        raise ValueError("image_augmentation must be an object with rotation_degrees, zoom, shift, intensity")
    settings = {**IMAGE_AUGMENTATION_DEFAULTS, **spec}
    limits = {"rotation_degrees": 180.0, "zoom": 1.0, "shift": 1.0, "intensity": 1.0}
    for name, value in settings.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= limits[name]):
            raise ValueError(f"image_augmentation {name} must be finite and in [0, {limits[name]}]")
    return settings


def image_kwargs(config: dict) -> dict:
    """The :func:`preprocessing.read_slot` settings for this rung."""
    low, high = config["slice_band"]
    return {
        "image_size": config["image_size"],
        "crop_mm": config["crop_mm"],
        "slices_per_plane": config["slices_per_plane"],
        "slice_band": (low, high),
    }


def soft_targets(lm_labels_csv: Path, uids: list[str], hard_masks: np.ndarray) -> np.ndarray:
    """The CSV's ``<finding>`` values for ``uids``, zeroed wherever the verdict is UNK."""
    frame = pd.read_csv(lm_labels_csv, dtype={UID: str}).set_index(UID)
    values = frame.loc[uids, LABELS].to_numpy(np.float32)
    if not np.isfinite(values[hard_masks > 0]).all():
        raise ValueError("Soft LM value missing where a verdict was observed")
    if ((values < 0) | (values > 1))[hard_masks > 0].any():
        raise ValueError("Soft LM values must lie in [0, 1]")
    return np.where(hard_masks > 0, values, 0.0).astype(np.float32)


def unk_supervision(lm_labels_csv: Path, uids: list[str], hard_masks: np.ndarray,
                    targets: np.ndarray, unk_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Targets and loss weights that also train UNK cells toward the CSV value at ``unk_weight``.

    Observed cells keep weight 1 and their target. UNK cells take the CSV's ``<finding>``
    value (0.28 for every UNK) at weight ``unk_weight``, the way pilkwang's labeler uses
    them, instead of being masked. Loss normalisation stays on the hard mask, so the
    observed cells' gradient is the parent rung's and the UNK term is purely additive.
    """
    frame = pd.read_csv(lm_labels_csv, dtype={UID: str}).set_index(UID)
    values = frame.loc[uids, LABELS].to_numpy(np.float32)
    unknown = hard_masks == 0
    if not np.isfinite(values[unknown]).all():
        raise ValueError("Soft LM value missing for an UNK verdict")
    weights = np.where(unknown, unk_weight, 1.0).astype(np.float32)
    return np.where(unknown, values, targets).astype(np.float32), weights


def label_columns(names: list[str]) -> list[int]:
    unknown = set(names) - set(LABELS)
    if unknown or not names:
        raise ValueError(f"Unknown or empty label list: {sorted(unknown)}")
    return [LABELS.index(name) for name in names]


def oversampled_rows(hard_y: np.ndarray, masks: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Training rows for one epoch: every study once, then ``copies`` extra draws of each
    study with an observed positive verdict on any of ``spec["labels"]``.

    Returns the rows and a per-row weight multiplier. With ``mask_other_labels`` the extra
    draws carry weight 0 on every other label, so those labels see exactly the base
    data and only the named labels gain repeats.
    """
    columns = label_columns(spec["labels"])
    positive = ((hard_y[:, columns] == 1) & (masks[:, columns] > 0)).any(1)
    extra = np.repeat(np.flatnonzero(positive), spec["copies"])
    rows = np.concatenate([np.arange(len(hard_y)), extra])
    keep = np.ones((len(rows), len(LABELS)), np.float32)
    if spec.get("mask_other_labels", True):
        others = [i for i in range(len(LABELS)) if i not in columns]
        keep[len(hard_y):, others] = 0.0
    return rows, keep


def label_weight_vector(weights: dict) -> np.ndarray:
    """Per-label loss weights, named labels at their value and the rest at 1, rescaled to
    mean 1 so the total loss scale (and with it the clipped gradient) matches the base."""
    vector = np.ones(len(LABELS), np.float32)
    for column, name in zip(label_columns(list(weights)), weights):
        vector[column] = weights[name]
    return vector / vector.mean()


def noisy_targets(targets: torch.Tensor, columns: list[int], sigma: float,
                  generator: torch.Generator) -> torch.Tensor:
    """Fresh Gaussian noise on ``columns`` of the soft targets, clipped back into [0, 1]."""
    noise = torch.randn(targets.shape[0], len(columns), generator=generator, device=targets.device)
    targets = targets.clone()
    targets[:, columns] = (targets[:, columns] + sigma * noise).clamp(0.0, 1.0)
    return targets


class TrainingRows(torch.utils.data.Dataset):
    """A dataset read at ``rows`` (repeats allowed); each item carries its position."""

    def __init__(self, dataset, rows):
        self.dataset, self.rows = dataset, [int(r) for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {**self.dataset[self.rows[index]], "row": index}


def dense_head(width: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    """ReLU MLP from ``width`` to one logit per label; exports as ``dense_v1``."""
    widths = [width, *hidden_dims, len(LABELS)]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class FinetuneModel(nn.Module):
    """Backbone plus dense head over the plane-averaged CLS + mean-patch feature."""

    def __init__(self, backbone: nn.Module, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = dense_head(2 * backbone.config.hidden_size, hidden_dims, dropout)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        # Must match submission.image_features, which serves the exported member.
        batch, planes = images.shape[:2]
        images = images.reshape(batch * planes, *images.shape[2:]).float()
        tokens = self.backbone(pixel_values=(images - self.mean) / self.std).last_hidden_state
        first_patch = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        pooled = torch.cat((tokens[:, 0], tokens[:, first_patch:].mean(1)), dim=-1)
        return pooled.reshape(batch, planes, -1).mean(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    def export_head(self) -> dict:
        return export_dense_head(self.head)


def export_dense_head(head: nn.Sequential) -> dict:
    """A :func:`dense_head` as ``dense_v1``; the identity scaler leaves features untouched."""
    width = head[0].in_features
    return {
        "kind": "dense_v1",
        "mean": torch.zeros(width),
        "scale": torch.ones(width),
        "layers": [
            {"weight": layer.weight.detach().float().cpu(), "bias": layer.bias.detach().float().cpu()}
            for layer in head
            if isinstance(layer, nn.Linear)
        ],
    }


class WindowModel(nn.Module):
    """R4: backbone over K 3-slice windows per plane, pooled under a validity mask.

    ``pooling="mean"`` (R4a) averages valid windows within each plane, then the
    planes, and feeds R1c's dense head. ``pooling="attention"`` (R4b) lets 12 learned
    label queries attend over all valid windows of all planes, each token carrying a
    learned plane embedding, and scores each label from its own pooled token. Keys
    are a low-dimensional projection and values are the raw tokens, which keeps the
    head's size near R1c's so R4b against R4a tests attention, not capacity.
    ``pooling="grouped"`` keeps slice position and plane identity: each plane's
    windows pool into ``window_groups`` contiguous runs, and the dense head reads all
    runs concatenated (:func:`submission.pool_windows`).
    """

    def __init__(self, backbone: nn.Module, hidden_dims: list[int], dropout: float, pooling: str,
                 key_dim: int = 128, window_groups: int | None = None):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        self.window_groups = window_groups if pooling == "grouped" else None
        width = 2 * backbone.config.hidden_size
        if pooling == "grouped" and not window_groups:
            raise ValueError("grouped pooling needs window_groups")
        if pooling in ("mean", "grouped"):
            self.head = dense_head(width * (len(PLANES) * window_groups if self.window_groups else 1),
                                   hidden_dims, dropout)
        else:
            self.plane_embedding = nn.Parameter(torch.zeros(len(PLANES), width))
            self.key = nn.Linear(width, key_dim)
            self.queries = nn.Parameter(torch.randn(len(LABELS), key_dim) / math.sqrt(key_dim))
            self.dropout = nn.Dropout(dropout)
            self.label_weight = nn.Parameter(torch.zeros(len(LABELS), width))
            self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))
            nn.init.normal_(self.label_weight, std=1 / math.sqrt(width))
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def tokens(self, images: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``[B, planes, K, width]`` CLS + mean-patch per window; invalid windows stay zero.

        Only valid windows go through the backbone.
        """
        flat = images.reshape(-1, *images.shape[-3:])[valid.reshape(-1)].float()
        hidden = self.backbone(pixel_values=(flat - self.mean) / self.std).last_hidden_state
        first_patch = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        pooled = torch.cat((hidden[:, 0], hidden[:, first_patch:].mean(1)), dim=-1)
        out = pooled.new_zeros(valid.numel(), pooled.shape[-1])
        out[valid.reshape(-1)] = pooled
        return out.reshape(*valid.shape, -1)

    def forward(self, images: torch.Tensor, valid: torch.Tensor, *, return_attention: bool = False):
        tokens = self.tokens(images, valid)
        if self.pooling in ("mean", "grouped"):
            logits, attention = self.head(pool_windows(tokens, valid, self.window_groups)), None
        else:
            tokens = (tokens + self.plane_embedding[None, :, None]).flatten(1, 2)
            scores = torch.einsum("lk,bnk->bln", self.queries, self.key(tokens)) / math.sqrt(self.queries.shape[1])
            scores = scores.float().masked_fill(~valid.flatten(1)[:, None], float("-inf"))
            attention = scores.softmax(-1).nan_to_num(0.0)
            pooled = self.dropout(torch.einsum("bln,bnd->bld", attention.to(tokens.dtype), tokens))
            logits = (pooled * self.label_weight).sum(-1) + self.label_bias
        return (logits, attention) if return_attention else logits


def load_backbone(config: dict) -> tuple[nn.Module, dict]:
    """The backbone to fine-tune, with its trainable parameters set, and its adaptation.

    Without ``init_checkpoint`` this is the pretrained model with its last blocks
    unfrozen. With it, the checkpoint's own adapted encoder is restored and its own
    adaptation (for example LoRA over the last blocks) stays trainable, so training
    continues exactly the parameters contrastive training adapted.
    """
    if config.get("init_checkpoint"):
        from .submission import load_encoder

        backbone, run_config, adaptation = load_encoder(ROOT / config["init_checkpoint"])
        if run_config.image_kwargs != image_kwargs(config):
            raise ValueError("init_checkpoint was trained on different image settings")
        return backbone.train(), adaptation
    if config.get("image_model_format") == "timm_dinov2":
        # e.g. OmniRad: timm weights renamed onto HF's Dinov2Model, config from the base.
        from .timm_dinov2 import load_timm_dinov2

        backbone = load_timm_dinov2(config["image_model"], config["image_model_revision"],
                                    base_model=config["image_model_base"],
                                    base_revision=config["image_model_base_revision"])
    elif config.get("image_model_format") == "timm":
        # e.g. CoAtNet: a timm convolutional/hybrid model behind the HF ViT interface.
        from .timm_backbone import load_timm_backbone

        backbone = load_timm_backbone(config["image_model"], config["image_model_revision"],
                                      grad_checkpointing=config.get("image_grad_checkpointing", False))
    else:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            config["image_model"],
            revision=config["image_model_revision"],
            local_files_only=True,
        )
    freeze_then_unfreeze_last(backbone, config["image_unfreeze_last"])
    adaptation = {"use_lora": False, "image_unfreeze_last": config["image_unfreeze_last"],
                  "lora_rank": 0, "lora_alpha": 0.0, "lora_dropout": 0.0}
    return backbone, adaptation


class RandomSliceDataset(torch.utils.data.Dataset):
    """Training studies with fresh jittered slices per plane on every read.

    Uncached by design: a cache would freeze the first draw. Each read seeds its own
    generator from torch's RNG, which the DataLoader seeds per worker from the
    loader's generator, so workers draw differently and a seed replays the run.

    Every epoch decodes DICOMs again, which is too slow over the WSL ``/mnt`` mount.
    ``mirror["to"]`` is a byte-identical local copy of the ``<study>/<series>`` tree,
    found by slot identity wherever the source tree is mounted. It is a host setting,
    filled in by :func:`with_host_mirror`.
    """

    def __init__(self, records, kwargs: dict, mirror: dict | None = None):
        self.records, self.kwargs, self.mirror = list(records), kwargs, mirror

    def _series(self, path: Path) -> Path:
        if not self.mirror:
            return path
        local = Path(self.mirror["to"]) / slot_identity(path)
        if not local.is_dir():
            raise FileNotFoundError(f"Series missing from the local mirror: {local}")
        return local

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rng = np.random.default_rng(int(torch.randint(2**62, (1,))))
        images = []
        for plane in PLANES:
            image = read_slot(self._series(record.slots[plane]), **self.kwargs, slice_rng=rng)
            if image is None:
                raise RuntimeError(f"DICOM decode failed after indexing: {record.study_uid} {plane}")
            images.append(apply_left_knee(image, plane, record.laterality))
        return {"images": torch.stack(images), "study_uid": record.study_uid}


class WindowDataset(RandomSliceDataset):
    """Studies as ``[planes, K, 3, H, W]`` windows plus a ``[planes, K]`` validity mask.

    Training draws jittered windows on every read. Evaluation takes stratum centres,
    which are deterministic, so they are cached in float16 under ``cache_dir``. An
    unreadable plane becomes a fully masked plane rather than an error.
    """

    def __init__(self, records, config: dict, *, training: bool):
        super().__init__(records, image_kwargs(config), config.get("train_series_mirror"))
        self.windows, self.training = config["windows_per_plane"], training
        self.cache_dir = None if training else ROOT / config["window_cache_dir"]
        self.settings = {"preprocessing_version": PREPROCESSING_VERSION, "windows": self.windows,
                         **{k: v for k, v in self.kwargs.items() if k != "slices_per_plane"}}

    def _read(self, record, rng) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.kwargs["image_size"]
        images = torch.zeros(len(PLANES), self.windows, 3, size, size)
        valid = torch.zeros(len(PLANES), self.windows, dtype=torch.bool)
        kwargs = {k: v for k, v in self.kwargs.items() if k != "slices_per_plane"}
        for i, plane in enumerate(PLANES):
            read = read_windows(self._series(record.slots[plane]), windows=self.windows,
                                **kwargs, slice_rng=rng)
            if read is not None:
                images[i], valid[i] = apply_left_knee_windows(*read, plane, record.laterality)
        return images, valid

    def cache_path(self, record) -> Path:
        key = sha256_text(json.dumps({**self.settings, "laterality": record.laterality,
                                      "slots": {p: slot_identity(record.slots[p]) for p in PLANES}},
                                     sort_keys=True))
        return self.cache_dir / f"{record.study_uid}-{key[:12]}.pt"

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if self.training:
            images, valid = self._read(record, np.random.default_rng(int(torch.randint(2**62, (1,)))))
            return {"images": images, "valid": valid, "study_uid": record.study_uid}
        path = self.cache_path(record)
        cached = None
        if path.exists():
            # A hard stop of WSL can keep write_atomic's rename but drop the data,
            # leaving an empty or truncated entry; rebuild it instead of crashing.
            try:
                cached = torch.load(path, weights_only=True)
            except Exception:  # any damage means rebuild; the entry is derived data
                cached = None
        if cached is not None:
            images, valid = cached["images"].float(), cached["valid"]
        else:
            images, valid = self._read(record, None)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            write_atomic(path, lambda t: torch.save({"images": images.half(), "valid": valid}, t))
        return {"images": images, "valid": valid, "study_uid": record.study_uid}

    def _series(self, path: Path) -> Path:
        # The mirror holds only training series. Evaluation reads fall back to the
        # source once, since their windows are then cached (see scripts/build_window_cache.py).
        try:
            return super()._series(path)
        except FileNotFoundError:
            if self.training:
                raise
            return path


def _dataset(records, config, *, training):
    if config.get("windows_per_plane"):
        return WindowDataset(records, config, training=training)
    # Evaluation, LM-test, gold and the exported member always see the fixed slices.
    if training and config.get("train_slice_sampling", "fixed") == "random":
        return RandomSliceDataset(records, image_kwargs(config), config.get("train_series_mirror"))
    return KneeReportDataset(records, image_kwargs(config), image_cache_dir=ROOT / config["image_cache_dir"])


def _loader(records, config, *, training, generator=None, rows=None):
    dataset = _dataset(records, config, training=training)
    if rows is not None:
        dataset = TrainingRows(dataset, rows)

    def collate(batch):
        # Model inputs first, as a tuple the model is called with; then the training row
        # positions, or the study uids when the dataset is read without rows.
        inputs = tuple(torch.stack([item[key] for item in batch])
                       for key in ("images", "valid") if key in batch[0])
        return inputs, [item["row"] if "row" in item else item["study_uid"] for item in batch]

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=training,
        generator=generator,
        num_workers=config["num_workers"],
        collate_fn=collate,
        # Evaluation workers respawn per pass: kept alive, they sat idle through
        # training and grew by ~0.5 GB each per pass until K=16 ran out of RAM.
        persistent_workers=training and config["num_workers"] > 0,
    )


def _to(inputs, device):
    return tuple(x.to(device, non_blocking=True) for x in inputs)


def augment_training_inputs(inputs: tuple, settings: dict | None,
                            generator: torch.Generator | None) -> tuple:
    """Fresh transforms per study/plane, shared across its channels and depth windows.

    Window channels are folded together before resampling so one coherent transform
    serves all K windows. Validity masks pass through, and invalid windows stay zero.
    Call only after device transfer in the training loop; cached pixels remain raw.
    """
    if settings is None:
        return inputs
    images = inputs[0]
    windowed = len(inputs) == 2
    grouped = images.flatten(-4, -3) if windowed else images
    augmented = augment_study_images(grouped, **settings, generator=generator).reshape_as(images)
    if windowed:
        augmented = augmented.masked_fill(~inputs[1][..., None, None, None], 0.0)
    return (augmented, *inputs[1:])


@torch.inference_mode()
def predict(model, loader, device, amp, *, extras: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Probabilities in loader order. For a windowed model, ``extras`` (a dict) collects
    the validity masks and, for attention pooling, the attention weights."""
    model.eval()
    parts, uids = [], []
    for inputs, batch_uids in loader:
        inputs = _to(inputs, device)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp):
            if extras is not None and isinstance(model, WindowModel):
                logits, attention = model(*inputs, return_attention=True)
                extras.setdefault("valid", []).append(inputs[1].cpu().numpy())
                if attention is not None:
                    extras.setdefault("attention", []).append(attention.float().cpu().numpy())
            else:
                logits = model(*inputs)
        parts.append(logits.float().sigmoid().cpu().numpy())
        uids.extend(batch_uids)
    return np.concatenate(parts), uids


def masked_window_counts(valid: np.ndarray) -> dict:
    """How often each fallback of design decision 6 fired, from ``[studies, planes, K]``."""
    windows = valid.shape[-1]
    return {
        "studies": int(valid.shape[0]),
        "windows": int(valid.size),
        "masked_windows": int((~valid).sum()),
        "masked_planes": int((~valid.any(-1)).sum()),
        "short_planes": int((valid.any(-1) & (valid.sum(-1) < windows)).sum()),
    }


def _scheduler(optimizer, total_steps: int, warmup_fraction: float):
    """Linear warmup into cosine decay to zero, stepped once per batch."""
    warmup = max(1, round(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def validate_config(config: dict) -> None:
    image_augmentation_settings(config)
    if config["targets"] not in TARGETS:
        raise ValueError(f"targets must be one of {TARGETS}")
    if config.get("train_slice_sampling", "fixed") not in SLICE_SAMPLING:
        raise ValueError(f"train_slice_sampling must be one of {SLICE_SAMPLING}")
    if config["image_unfreeze_last"] < 1:
        raise ValueError("Fine-tuning needs at least one unfrozen block")
    if config["slices_per_plane"] != 3:
        raise ValueError("The image path takes exactly 3 slices per plane")
    if config.get("windows_per_plane") is not None:
        if config["windows_per_plane"] < 1:
            raise ValueError("windows_per_plane must be positive")
        if config.get("pooling") not in POOLING:
            raise ValueError(f"A windowed rung needs pooling in {POOLING}")
        if "window_cache_dir" not in config:
            raise ValueError("A windowed rung needs window_cache_dir for evaluation windows")
        if (config["pooling"] == "grouped") != ("window_groups" in config):
            raise ValueError("window_groups goes with grouped pooling, and only with it")
        if config["pooling"] == "grouped" and (
                config["window_groups"] < 1 or config["windows_per_plane"] % config["window_groups"]):
            raise ValueError("window_groups must divide windows_per_plane")
    elif "pooling" in config:
        raise ValueError("pooling applies only with windows_per_plane")
    if not 0 <= config.get("unk_weight", 0) <= 1:
        raise ValueError("unk_weight must lie in [0, 1]")
    if config.get("grad_accumulation", 1) < 1:
        raise ValueError("grad_accumulation must be positive")
    for key in ("epochs", "batch_size", "patience"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if "oversample" in config:
        label_columns(config["oversample"]["labels"])
        if config["oversample"]["copies"] < 1:
            raise ValueError("oversample copies must be positive")
    if "label_noise" in config:
        label_columns(config["label_noise"]["labels"])
        if not config["label_noise"]["sigma"] > 0:
            raise ValueError("label_noise sigma must be positive")
    if "divergence_stop" in config:
        if not config["divergence_stop"]["margin"] > 0 or config["divergence_stop"]["epochs"] < 1:
            raise ValueError("divergence_stop needs a positive margin and epochs")
    if "resume_from" in config and not config["resume_from"]["lr_scale"] > 0:
        raise ValueError("resume_from needs a positive lr_scale")
    if "label_weights" in config:
        label_columns(list(config["label_weights"]))
        if any(w <= 0 for w in config["label_weights"].values()):
            raise ValueError("label_weights must be positive")


def with_host_mirror(config: dict, paths_config: Path | None = None) -> dict:
    """Name this machine's ``train_series`` mirror in a rung's configuration.

    Where the mirror lives is a property of the host, not of the experiment, so
    committed rungs leave ``train_series_mirror`` out and ``paths.json`` supplies it.
    A rung that still states one (an older run's ``config.json``) keeps it. Called by
    entry points, so the run's own ``config.json`` records the concrete path.
    """
    if config.get("train_series_mirror") is not None:
        return config
    mirror = paths.load_settings(paths_config).get("train_series_mirror")
    return config if mirror is None else {**config, "train_series_mirror": {"to": str(mirror)}}


def run_finetune(config: dict, output: Path, *, device: str = "cuda") -> dict:
    """Train one rung, select its epoch on validation, then score LM-test and gold once."""
    validate_config(config)
    augmentation = image_augmentation_settings(config)
    if augmentation is not None:
        config = {**config, "image_augmentation": augmentation}
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)

    groups, targets, masks, population = load_population(config, ROOT)
    lm_csv = ROOT / config["lm_labels_csv"]
    hard = {split: targets[split] for split in ("train", "validation")}
    if config["targets"] == "soft":
        train_y = soft_targets(lm_csv, [r.study_uid for r in groups["train"]], masks["train"])
    else:
        train_y = hard["train"]
    train_w = masks["train"]
    if config.get("unk_weight"):
        train_y, train_w = unk_supervision(lm_csv, [r.study_uid for r in groups["train"]], masks["train"],
                                           train_y, config["unk_weight"])
    # Rows index the training tables. Without oversampling each study is read once, as before.
    epoch_rows = np.arange(len(groups["train"]))
    if config.get("oversample"):
        epoch_rows, keep = oversampled_rows(hard["train"], masks["train"], config["oversample"])
        train_y, train_w = train_y[epoch_rows], train_w[epoch_rows] * keep
    clean_w = train_w
    if config.get("label_weights"):
        train_w = train_w * label_weight_vector(config["label_weights"])

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    backbone, adaptation = load_backbone(config)
    windowed = bool(config.get("windows_per_plane"))
    if windowed:
        model = WindowModel(backbone, config["head_hidden"], config["dropout"], config["pooling"],
                            window_groups=config.get("window_groups"))
    else:
        model = FinetuneModel(backbone, config["head_hidden"], config["dropout"])
    model = model.to(device)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config["lr_backbone"]},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")],
             "lr": config["lr_head"]},
        ],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator().manual_seed(config["seed"])
    augmentation_generator = (torch.Generator(device=device).manual_seed(config["seed"] + 2)
                              if augmentation is not None else None)
    train_loader = _loader(groups["train"], config, training=True, generator=generator, rows=epoch_rows)
    validation_loader = _loader(groups["validation"], config, training=False)
    accumulation = config.get("grad_accumulation", 1)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    scaler = torch.amp.GradScaler(enabled=config["amp"])
    resume = config.get("resume_from")
    start_epoch = 0
    if resume:
        # Warm restart: weights, optimizer moments and AMP scale from the parent's last.pt,
        # then a fresh warmup + cosine over the remaining epochs at lr_scale x the base rates.
        parent = ROOT / resume["run"]
        last = torch.load(parent / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(last["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["scaler"])
        if augmentation_generator is not None and "augmentation_rng_state" in last:
            augmentation_generator.set_state(last["augmentation_rng_state"].cpu())
        for group, base in zip(optimizer.param_groups, (config["lr_backbone"], config["lr_head"])):
            group["lr"] = base * resume["lr_scale"]
            group.pop("initial_lr", None)
        start_epoch = last["epoch"]
        if start_epoch >= config["epochs"]:
            raise ValueError("resume_from parent already reached the epoch budget")
    scheduler = _scheduler(optimizer, (config["epochs"] - start_epoch) * steps_per_epoch, config["warmup_fraction"])
    y_all = torch.as_tensor(train_y, device=device)
    m_all = torch.as_tensor(train_w, device=device)
    # The logged training BCE stays unweighted and noise-free, comparable with the base rung.
    m_clean = torch.as_tensor(clean_w, device=device)
    noise = config.get("label_noise")
    if noise:
        noise_columns = label_columns(noise["labels"])
        noise_generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    # Fixed full-split observation rates keep minibatch losses on one macro scale.
    # They come from the hard mask, so UNK weight adds to a label's loss without
    # rescaling its observed cells.
    observed_rates = torch.as_tensor(masks["train"], device=device).mean(0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    print(
        json.dumps({"population": population["coverage"], "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad)}),
        flush=True,
    )

    best = {"validation_masked_bce": float("inf"), "epoch": 0, "state": None}
    if resume:
        # The parent's history and best checkpoint carry over: the extension must beat it.
        (output / "history.jsonl").write_text((parent / "history.jsonl").read_text())
        parent_best = torch.load(parent / "best.pt", map_location="cpu", weights_only=False)
        best = {"validation_masked_bce": read_json(parent / "report.json")["validation_masked_bce"],
                "epoch": parent_best["epoch"], "state": parent_best["trainable_state_dict"]}
    stale, above, stopped = 0, 0, "max_epochs"
    divergence = config.get("divergence_stop")
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        model.train()
        epoch_started, losses, objectives, train_valid = time.perf_counter(), [], [], []
        optimizer.zero_grad(set_to_none=True)
        for batch, (inputs, batch_rows) in enumerate(train_loader, start=1):
            index = torch.as_tensor(batch_rows, device=device)
            training_inputs = augment_training_inputs(_to(inputs, device), augmentation, augmentation_generator)
            with torch.autocast(device_type=device.split(":")[0], enabled=config["amp"]):
                logits = model(*training_inputs)
            targets_batch = y_all[index]
            if noise:
                targets_batch = noisy_targets(targets_batch, noise_columns, noise["sigma"], noise_generator)
            loss = masked_bce(logits.float(), targets_batch, m_all[index], observed_rates)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss / accumulation).backward()
            objectives.append(loss.item())
            losses.append(float(masked_bce(logits.detach().float(), y_all[index], m_clean[index], observed_rates)))
            if windowed:
                train_valid.append(inputs[1].numpy())
            # Step every `accumulation` micro-batches, and flush a partial group at epoch end.
            if batch % accumulation == 0 or batch == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if config.get("max_steps") and batch >= config["max_steps"]:
                break
        if windowed and epoch == start_epoch + 1:
            window_counts = {"train": masked_window_counts(np.concatenate(train_valid))}
        probabilities, uids = predict(model, validation_loader, device, config["amp"])
        order = {u: i for i, u in enumerate(uids)}
        probabilities = probabilities[[order[r.study_uid] for r in groups["validation"]]]
        logits = torch.logit(torch.as_tensor(probabilities).clamp(1e-6, 1 - 1e-6))
        validation_bce = float(
            masked_bce(logits, torch.as_tensor(hard["validation"]), torch.as_tensor(masks["validation"]))
        )
        validation_auroc = masked_macro_auroc(probabilities, hard["validation"], masks["validation"])
        row = {
            "epoch": epoch,
            "train_masked_bce": float(np.mean(losses)),
            **({"train_objective": float(np.mean(objectives))}
               if noise or config.get("label_weights") else {}),
            "validation_masked_bce": validation_bce,
            "validation_macro_auroc": validation_auroc["macro_auroc"],
            "seconds": time.perf_counter() - epoch_started,
        }
        if device.startswith("cuda"):
            row["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row), flush=True)
        if validation_bce < best["validation_masked_bce"] - config["min_delta"]:
            best = {
                "validation_masked_bce": validation_bce,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k in trainable},
            }
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                stopped = "patience"
                break
        # Divergence stop: end only once validation BCE sits clearly above its best for
        # several epochs in a row, rather than after a run of epochs that merely fail to improve.
        if divergence:
            above = above + 1 if validation_bce > best["validation_masked_bce"] + divergence["margin"] else 0
            if above >= divergence["epochs"]:
                stopped = "divergence"
                break

    if config.get("save_last"):
        # The final epoch's full training state, so a later run can extend this one.
        last = {"config": config, "epoch": epoch,
                "trainable_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
        if augmentation_generator is not None:
            last["augmentation_rng_state"] = augmentation_generator.get_state()
        write_atomic(output / "last.pt", lambda t: torch.save(last, t))
    model.load_state_dict(best["state"], strict=False)
    write_atomic(output / "best.pt", lambda t: torch.save(
        {"config": config, "epoch": best["epoch"], "trainable_state_dict": best["state"]}, t))

    # Scoring happens once, after the epoch choice is frozen.
    test_rows, test_y, test_m, test_identity = load_test_population(
        ROOT / config["cohort_manifest"], lm_csv, ROOT / config["gold_labels_csv"]
    )
    scored, extras = {}, {}
    for name, rows in (("validation", groups["validation"]), ("lm_test", test_rows), ("gold", groups["gold"])):
        if name == "validation" and not windowed:
            continue
        extras[name] = {} if windowed else None
        loader = _loader(rows, config, training=False)
        probabilities, uids = predict(model, loader, device, config["amp"], extras=extras[name])
        order = [{u: i for i, u in enumerate(uids)}[r.study_uid] for r in rows]
        scored[name] = probabilities[order]
        if windowed:
            valid = np.concatenate(extras[name]["valid"])[order]
            window_counts[name] = masked_window_counts(valid)
            extras[name] = {"valid": valid, **({"attention": np.concatenate(extras[name]["attention"])[order]}
                                               if "attention" in extras[name] else {})}
    lm_test = masked_macro_auroc(scored["lm_test"], test_y, test_m)
    gold = score_predictions(targets["gold"], scored["gold"])

    def save_predictions(path):
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                lm_test_uids=np.asarray(test_identity["study_uids"]),
                lm_test=scored["lm_test"],
                gold_uids=np.asarray([r.study_uid for r in groups["gold"]]),
                gold=scored["gold"],
                gold_targets=targets["gold"],
            )

    write_atomic(output / "predictions.npz", save_predictions)

    if windowed:
        # Validity masks and, for attention pooling, [studies, labels, planes*K] weights.
        def save_windows(path):
            with path.open("wb") as handle:
                np.savez_compressed(handle, lm_test_uids=np.asarray(test_identity["study_uids"]),
                                    **{f"lm_test_{k}": v for k, v in extras["lm_test"].items()})

        write_atomic(output / "windows.npz", save_windows)
    else:
        # A Kaggle bundle member, loadable by submission.load_member. The Kaggle path
        # serves 3 slices per plane, so windowed rungs are not exported (design decision 10).
        export_member(
            output / "member-00.pt",
            model.backbone,
            SimpleNamespace(image_kwargs=image_kwargs(config)),
            adaptation,
            model.export_head(),
        )
    report = {
        "rung": config["name"],
        **({"image_augmentation": {**augmentation, "shared_across_windows": True,
                                   "seed": config["seed"] + 2}} if augmentation is not None else {}),
        "best_epoch": best["epoch"],
        "stopped": stopped,
        "validation_masked_bce": best["validation_masked_bce"],
        "lm_test": {**lm_test, "studies": len(test_y),
                    "manifest_sha256": test_identity["manifest_sha256"]},
        "gold": {"macro_auroc": gold["macro_auroc"], "per_label": gold["per_label"],
                 "studies": len(targets["gold"])},
        "population": population,
        "lm_labels_sha256": sha256_file(lm_csv),
        **({"unk_weight": config["unk_weight"], "unk_weight_share": dict(zip(LABELS, (
            (train_w.sum(0) - masks["train"].sum(0)) / masks["train"].sum(0)).round(4).tolist()))}
           if config.get("unk_weight") else {}),
        **({"oversample": {"draws_per_epoch": len(epoch_rows), "extra_draws": len(epoch_rows) - len(groups["train"]),
                           "observed_draws": dict(zip(LABELS, (clean_w > 0).sum(0).tolist()))}}
           if config.get("oversample") else {}),
        **({"label_weights": dict(zip(LABELS, label_weight_vector(config["label_weights"]).round(4).tolist()))}
           if config.get("label_weights") else {}),
        **({"window_counts": window_counts, "exported": False} if windowed else {}),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    print(f"{config['name']}: LM-test macro AUROC={lm_test['macro_auroc']:.4f}, "
          f"gold macro AUROC={gold['macro_auroc']:.4f}, epoch={best['epoch']}", flush=True)
    return report
