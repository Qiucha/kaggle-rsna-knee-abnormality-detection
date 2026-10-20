from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from contrastive_learning.cli import build_parser, config_from_args
from contrastive_learning.config import TrainConfig
from contrastive_learning.declip import (
    ReportEmbeddingQueue,
    augment_report_tokens,
    compute_declip_loss,
)


def test_augment_report_tokens_preserves_special_tokens_and_padding() -> None:
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    input_ids[:, 0] = 101  # CLS token
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[:, 12:] = 0  # padding tokens

    mask_token_id = 999
    aug_ids = augment_report_tokens(
        input_ids, attention_mask, mask_token_id=mask_token_id, mask_prob=0.3
    )

    assert aug_ids.shape == input_ids.shape
    assert aug_ids.dtype == input_ids.dtype
    # CLS token preserved
    assert (aug_ids[:, 0] == 101).all()
    # Padding tokens untouched
    assert (aug_ids[:, 12:] == input_ids[:, 12:]).all()
    # At least some non-special tokens masked
    assert (aug_ids[:, 1:12] == mask_token_id).sum() > 0


def test_augment_report_tokens_with_zero_prob() -> None:
    input_ids = torch.randint(10, 100, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    aug_ids = augment_report_tokens(input_ids, attention_mask, mask_token_id=999, mask_prob=0.0)
    assert torch.equal(aug_ids, input_ids)


def test_report_embedding_queue_fifo_and_retrieval() -> None:
    capacity = 8
    dim = 16
    queue = ReportEmbeddingQueue(capacity=capacity, embedding_dim=dim)
    assert queue.size == 0

    # In-batch fallback when queue empty
    batch_1 = F.normalize(torch.randn(4, dim), dim=-1)
    nn_1 = queue.get_nearest_neighbours(batch_1)
    assert nn_1.shape == (4, dim)

    # Enqueue batch_1
    queue.enqueue(batch_1)
    assert queue.size == 4

    # Enqueue batch_2
    batch_2 = F.normalize(torch.randn(6, dim), dim=-1)
    queue.enqueue(batch_2)
    assert queue.size == capacity
    assert queue.is_full

    # Retrieval from queue
    query = F.normalize(torch.randn(3, dim), dim=-1)
    nn_query = queue.get_nearest_neighbours(query)
    assert nn_query.shape == (3, dim)


def test_compute_declip_loss_backward_and_breakdown() -> None:
    B, D = 4, 16
    z_I = torch.randn(B, D, requires_grad=True)
    z_I_aug = torch.randn(B, D, requires_grad=True)
    z_T = torch.randn(B, D, requires_grad=True)
    z_T_aug = torch.randn(B, D, requires_grad=True)
    z_T_nn = torch.randn(B, D)

    # Normalize
    z_I_n = F.normalize(z_I, dim=-1)
    z_I_aug_n = F.normalize(z_I_aug, dim=-1)
    z_T_n = F.normalize(z_T, dim=-1)
    z_T_aug_n = F.normalize(z_T_aug, dim=-1)
    z_T_nn_n = F.normalize(z_T_nn, dim=-1)

    logit_scale = torch.tensor(2.65, requires_grad=True)

    loss, breakdown = compute_declip_loss(
        z_I=z_I_n,
        z_I_aug=z_I_aug_n,
        z_T=z_T_n,
        z_T_aug=z_T_aug_n,
        z_T_nn=z_T_nn_n,
        logit_scale=logit_scale,
        mv_weight=1.0,
        nn_weight=0.5,
        image_ss_weight=0.5,
    )

    assert loss.ndim == 0
    assert "loss_mv" in breakdown
    assert "loss_nn" in breakdown
    assert "loss_img_ss" in breakdown
    assert breakdown["loss_mv"] > 0
    assert breakdown["loss_nn"] > 0
    assert breakdown["loss_img_ss"] > 0

    loss.backward()
    assert z_I.grad is not None
    assert z_I_aug.grad is not None
    assert z_T.grad is not None
    assert z_T_aug.grad is not None


def test_declip_cli_parsing_and_config_validation() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--use-declip",
        "--declip-mv-weight", "1.2",
        "--declip-nn-weight", "0.4",
        "--declip-image-ss-weight", "0.6",
        "--declip-queue-size", "256",
        "--declip-text-mask-prob", "0.20",
    ])
    assert args.use_declip is True
    assert args.declip_mv_weight == 1.2
    assert args.declip_nn_weight == 0.4
    assert args.declip_image_ss_weight == 0.6
    assert args.declip_queue_size == 256
    assert args.declip_text_mask_prob == 0.20

    # Config validation
    valid_config = TrainConfig(
        use_declip=True,
        declip_mv_weight=1.0,
        declip_nn_weight=0.5,
        declip_image_ss_weight=0.5,
        declip_queue_size=128,
        declip_text_mask_prob=0.15,
    )
    valid_config.validate_training()

    # Invalid queue size
    invalid_config = TrainConfig(use_declip=True, declip_queue_size=1)
    with pytest.raises(ValueError, match="declip_queue_size must be at least 2"):
        invalid_config.validate_training()

    # Invalid negative weight
    invalid_weight = TrainConfig(use_declip=True, declip_mv_weight=-0.5)
    with pytest.raises(ValueError, match="DeCLIP loss weights cannot be negative"):
        invalid_weight.validate_training()
