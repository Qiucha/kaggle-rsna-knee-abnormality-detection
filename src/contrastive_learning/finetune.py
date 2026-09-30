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

UNK verdicts are masked in every rung. ``targets: "soft"`` swaps only the observed
cells' 0/1 for the CSV's ``<finding>`` value. That value is a lookup from verdict and
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

from .artifacts import sha256_file, sha256_text, write_atomic, write_json
from .data import KneeReportDataset
from .lm_proxy import load_test_population, masked_macro_auroc
from .lm_supervision import load_population
from .preprocessing import (
    PLANES,
    PREPROCESSING_VERSION,
    apply_left_knee,
    apply_left_knee_windows,
    read_slot,
    read_windows,
)
from .model import _IMAGENET_MEAN, _IMAGENET_STD, freeze_then_unfreeze_last
from .submission import LABELS, UID, export_member, score_predictions
from .supervised_head import masked_bce

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("verdict", "soft")
SLICE_SAMPLING = ("fixed", "random")
POOLING = ("mean", "attention")


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
        """The head as ``dense_v1``; the identity scaler leaves features untouched."""
        width = self.head[0].in_features
        return {
            "kind": "dense_v1",
            "mean": torch.zeros(width),
            "scale": torch.ones(width),
            "layers": [
                {"weight": layer.weight.detach().float().cpu(), "bias": layer.bias.detach().float().cpu()}
                for layer in self.head
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
    """

    def __init__(self, backbone: nn.Module, hidden_dims: list[int], dropout: float, pooling: str,
                 key_dim: int = 128):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        width = 2 * backbone.config.hidden_size
        if pooling == "mean":
            self.head = dense_head(width, hidden_dims, dropout)
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
        weights = valid.to(tokens.dtype)
        if self.pooling == "mean":
            counts = weights.sum(-1)
            planes = (tokens * weights[..., None]).sum(2) / counts.clamp(min=1)[..., None]
            present = (counts > 0).to(tokens.dtype)
            study = (planes * present[..., None]).sum(1) / present.sum(1).clamp(min=1)[:, None]
            logits, attention = self.head(study), None
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
    ``mirror`` maps a series-root prefix to a byte-identical local copy.
    """

    def __init__(self, records, kwargs: dict, mirror: dict | None = None):
        self.records, self.kwargs, self.mirror = list(records), kwargs, mirror

    def _series(self, path: Path) -> Path:
        if not self.mirror:
            return path
        local = Path(self.mirror["to"]) / path.relative_to(self.mirror["from"])
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

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if self.training:
            images, valid = self._read(record, np.random.default_rng(int(torch.randint(2**62, (1,)))))
            return {"images": images, "valid": valid, "study_uid": record.study_uid}
        key = sha256_text(json.dumps({**self.settings, "laterality": record.laterality,
                                      "slots": {p: str(record.slots[p]) for p in PLANES}}, sort_keys=True))
        path = self.cache_dir / f"{record.study_uid}-{key[:12]}.pt"
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


def _loader(records, config, *, training, generator=None):
    dataset = _dataset(records, config, training=training)

    def collate(batch):
        # Model inputs first, as a tuple the model is called with; study uids last.
        inputs = tuple(torch.stack([item[key] for item in batch])
                       for key in ("images", "valid") if key in batch[0])
        return inputs, [item["study_uid"] for item in batch]

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
    elif "pooling" in config:
        raise ValueError("pooling applies only with windows_per_plane")
    if config.get("grad_accumulation", 1) < 1:
        raise ValueError("grad_accumulation must be positive")
    for key in ("epochs", "batch_size", "patience"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")


def run_finetune(config: dict, output: Path, *, device: str = "cuda") -> dict:
    """Train one rung, select its epoch on validation, then score LM-test and gold once."""
    validate_config(config)
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
    train_rows = {r.study_uid: i for i, r in enumerate(groups["train"])}

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    backbone, adaptation = load_backbone(config)
    windowed = bool(config.get("windows_per_plane"))
    if windowed:
        model = WindowModel(backbone, config["head_hidden"], config["dropout"], config["pooling"])
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
    train_loader = _loader(groups["train"], config, training=True, generator=generator)
    validation_loader = _loader(groups["validation"], config, training=False)
    accumulation = config.get("grad_accumulation", 1)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = _scheduler(optimizer, config["epochs"] * steps_per_epoch, config["warmup_fraction"])
    scaler = torch.amp.GradScaler(enabled=config["amp"])
    y_all = torch.as_tensor(train_y, device=device)
    m_all = torch.as_tensor(masks["train"], device=device)
    # Fixed full-split observation rates keep minibatch losses on one macro scale.
    observed_rates = m_all.mean(0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    print(
        json.dumps({"population": population["coverage"], "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad)}),
        flush=True,
    )

    best = {"validation_masked_bce": float("inf"), "epoch": 0, "state": None}
    stale = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_started, losses, train_valid = time.perf_counter(), [], []
        optimizer.zero_grad(set_to_none=True)
        for batch, (inputs, uids) in enumerate(train_loader, start=1):
            index = torch.as_tensor([train_rows[u] for u in uids], device=device)
            with torch.autocast(device_type=device.split(":")[0], enabled=config["amp"]):
                logits = model(*_to(inputs, device))
            loss = masked_bce(logits.float(), y_all[index], m_all[index], observed_rates)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss / accumulation).backward()
            losses.append(float(loss))
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
        if windowed and epoch == 1:
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
                break

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
        "best_epoch": best["epoch"],
        "validation_masked_bce": best["validation_masked_bce"],
        "lm_test": {**lm_test, "studies": len(test_y),
                    "manifest_sha256": test_identity["manifest_sha256"]},
        "gold": {"macro_auroc": gold["macro_auroc"], "per_label": gold["per_label"],
                 "studies": len(targets["gold"])},
        "population": population,
        "lm_labels_sha256": sha256_file(lm_csv),
        **({"window_counts": window_counts, "exported": False} if windowed else {}),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    print(f"{config['name']}: LM-test macro AUROC={lm_test['macro_auroc']:.4f}, "
          f"gold macro AUROC={gold['macro_auroc']:.4f}, epoch={best['epoch']}", flush=True)
    return report
