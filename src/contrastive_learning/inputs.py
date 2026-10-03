"""Resolve the same studies, splits, and loaders for training and evaluation.

Training and offline checkpoint evaluation must agree exactly on which studies are
in a split and in what order, or their numbers are not comparable. Rather than
trusting two call sites to stay in step, both go through :func:`load_cohort` and
:func:`make_loader` here; the only differences between them are the two named
arguments this module exposes, ``for_training`` and ``training``.

Loading is separated from splitting on purpose. Indexing a cohort is expensive and
can be inspected on its own (``--prepare-only``), and evaluation asks for a single
split without needing the others to be derivable.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_learning.config import MULTILINGUAL_TEXT_MODELS, TrainConfig
from contrastive_learning.data import (
    KneeReportDataset,
    StudyRecord,
    build_study_records,
    make_collator,
    split_records,
    split_records_fold,
)
from contrastive_learning.manifest import read_cohort_manifest


SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class StudyCohort:
    """Loaded studies plus the one split policy that applies to them.

    Splits are computed on first request and cached, so repeated access returns the
    identical lists; merely loading a cohort never commits to a split.
    """

    records: list[StudyRecord]
    summary: dict[str, Any]
    _config: TrainConfig = field(repr=False)
    _splits: dict[str, list[StudyRecord]] | None = field(default=None, init=False, repr=False)

    def split(self, name: str) -> list[StudyRecord]:
        """Return one split: read from the manifest, or derived from the seed.

        A frozen cohort carries its assignment per study, which is the only way to
        get a held-out test split. The legacy CSV path can only produce train and
        validation, from either a seeded holdout fraction or a seeded K-fold.
        """
        if name not in SPLIT_NAMES:
            raise ValueError(f"Unknown cohort split: {name!r}")
        config = self._config
        if config.cohort_manifest is not None:
            return [record for record in self.records if record.split == name]
        if name == "test":
            raise ValueError("Legacy checkpoints only have train and validation splits")
        if self._splits is None:
            if config.fold_count is None:
                train, validation = split_records(self.records, config.val_fraction, config.seed)
            else:
                train, validation = split_records_fold(
                    self.records, config.fold_count, config.fold_index, config.seed
                )
            self._splits = {"train": train, "validation": validation}
        return self._splits[name]


def load_cohort(config: TrainConfig, *, for_training: bool = False) -> StudyCohort:
    """Load the studies a run operates on, from a frozen manifest or from CSVs.

    ``for_training`` enables the two checks that only make sense before training:
    the cohort's text policy must match the text encoder, and pixel validation is
    honoured. Evaluating a saved checkpoint skips both -- its cohort was already
    validated when the run was created, and re-decoding it would only cost time.
    """
    if config.cohort_manifest is not None:
        series_root = config.series_root if config.data_root is not None else None
        records, metadata = read_cohort_manifest(config.cohort_manifest, series_root=series_root)
        text_policy = metadata["text_policy"]
        if (
            for_training
            and text_policy == "multilingual_raw"
            and config.text_model not in MULTILINGUAL_TEXT_MODELS
            and not config.allow_text_policy_mismatch
        ):
            raise ValueError(
                "multilingual_raw manifest requires a registered multilingual text model; "
                "use multilingual-e5 or explicitly pass --allow-text-policy-mismatch"
            )
        # Truncating before splitting keeps a smoke run's splits a prefix of the
        # real cohort's, rather than a different partition of a smaller set.
        if config.max_studies is not None:
            records = records[: config.max_studies]
        summary = {
            "manifest": str(config.cohort_manifest),
            "text_policy": text_policy,
            "selected": len(records),
            **{name: sum(record.split == name for record in records) for name in SPLIT_NAMES},
        }
    else:
        records, data_summary = build_study_records(
            config.reports_csv,
            config.series_csv,
            config.series_root,
            report_column=config.report_column,
            validate_pixels=for_training and config.validate_pixels,
            image_kwargs=config.image_kwargs,
            max_studies=config.max_studies,
        )
        summary = asdict(data_summary)
    return StudyCohort(records, summary, config)


def _seed_worker(worker_id: int) -> None:
    """Give each loader worker a distinct, run-reproducible random stream.

    PyTorch seeds its own generator per worker but leaves ``random`` and NumPy
    identical across them, which would repeat the same augmentation draws.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader(
    records: list[StudyRecord],
    tokenizer: Any,
    config: TrainConfig,
    *,
    training: bool = False,
    generator: torch.Generator | None = None,
    label_table: Any | None = None,
) -> DataLoader:
    """Build paired batches of studies and tokenized reports.

    Only training shuffles and drops the final partial batch: contrastive loss is
    computed within a batch, so a short last batch would be scored against fewer
    negatives than the rest. Evaluation keeps every study, in cohort order, so
    retrieval runs against the complete candidate pool.

    Pass the training ``generator`` so a resumed run continues the same shuffle.

    ``label_table`` is supplied only for training: it adds the LM finding vectors
    to each batch so the loss can soften its target. Evaluation leaves it None and
    is scored on the unmodified diagonal, which is what keeps retrieval metrics
    comparable between runs trained with and without it.
    """
    return DataLoader(
        KneeReportDataset(
            records,
            config.image_kwargs,
            image_cache_dir=config.image_cache_dir,
            memory_cache_size=config.memory_cache_size,
            label_table=label_table,
        ),
        batch_size=config.batch_size,
        shuffle=training,
        drop_last=training,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=make_collator(
            tokenizer, config.max_text_length, config.text_prefix, config.report_view
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
    )
