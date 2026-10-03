"""DeCLIP (Data-Efficient CLIP) components for low-data image-report alignment.

Based on docs/notes/Training_Guide_on_CLIP.md and DeCLIP / Windsor architectures:
1. Multi-View Cross-Modal Supervision (L_MV):
   Four cross-modal pairs (X, R), (X', R), (X, R'), (X', R') under bidirectional
   InfoNCE loss, where X' is an augmented MRI view preserving knee anatomy and
   R' is a masked/perturbed report preserving clinical findings.

2. Nearest-Neighbour Report Supervision (L_NN):
   A FIFO memory queue Q_T of report embeddings. For each report R_i, retrieves
   its nearest neighbour R_i^NN = argmax_{j != i} (z_i^T)^T z_j^T and adds
   (X_i, R_i^NN) and (X_i', R_i^NN) as additional positive pairs.

3. Unimodal Self-Supervision (L_I, L_T):
   SimCLR-style image contrastive loss between physical views X and X'.

4. Total Training Objective:
   L_total = L_MV + lambda_NN * L_NN + lambda_I * L_I
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from contrastive_learning.model import (
    soft_target_contrastive_loss,
    symmetric_contrastive_loss,
)


def augment_report_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int | None,
    mask_prob: float = 0.15,
    special_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Randomly mask tokens in report sequence, preserving special tokens and padding.

    Clinical findings and anatomical descriptions are partially masked with mask tokens
    (default 15% rate, following DeCLIP and MLM standards), generating a perturbed
    report view R' while preserving syntax structure and clinical semantic framing.
    """
    if mask_token_id is None or mask_prob <= 0.0:
        return input_ids

    cand_mask = attention_mask.bool()
    if cand_mask.shape[1] > 0:
        cand_mask[:, 0] = False  # Protect CLS/BOS token at index 0

    if special_token_ids:
        for st_id in special_token_ids:
            cand_mask = cand_mask & (input_ids != st_id)

    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask = cand_mask & (rand < mask_prob)

    # If all candidate tokens for a sample were masked, keep at least one unmasked
    all_masked = (mask & cand_mask).sum(dim=-1) == cand_mask.sum(dim=-1)
    if all_masked.any():
        mask[all_masked] = False

    aug_ids = input_ids.clone()
    aug_ids[mask] = mask_token_id
    return aug_ids


class ReportEmbeddingQueue:
    """FIFO queue of report embeddings for DeCLIP nearest-neighbour supervision.

    Maintains a rolling memory bank of L2-normalized report embeddings. For each
    report in the current batch, retrieves the most semantically similar report
    embedding j* = argmax_{j != i} (z_i^T)^T z_j^T to serve as an additional
    positive pairing target (X_i, R_i^{NN}) and (X_i', R_i^{NN}).
    """

    def __init__(
        self,
        capacity: int = 512,
        embedding_dim: int = 256,
        device: torch.device | None = None,
    ):
        if capacity < 2:
            raise ValueError("ReportEmbeddingQueue capacity must be at least 2")
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cpu")
        self.queue: torch.Tensor | None = None
        self.ptr: int = 0
        self.is_full: bool = False

    @property
    def size(self) -> int:
        if self.queue is None:
            return 0
        return self.capacity if self.is_full else self.ptr

    def get_nearest_neighbours(self, z_T: torch.Tensor) -> torch.Tensor:
        """Find the nearest neighbour report embedding for each z_T[i].

        Returns tensor of shape [B, D] where row i is the NN embedding for z_T[i].
        Falls back to in-batch nearest neighbour if the queue is empty or has fewer
        items than the batch size.
        """
        B, D = z_T.shape
        device = z_T.device

        # In-batch candidate similarities: [B, B]
        in_batch_sim = z_T @ z_T.T
        # Mask out self on the diagonal
        in_batch_sim = in_batch_sim.clone()
        in_batch_sim.fill_diagonal_(-float("inf"))

        if self.size < B:
            nn_indices = in_batch_sim.argmax(dim=-1)
            return z_T[nn_indices]

        # Queue candidate similarities: [B, K]
        queue_tensor = self.queue[: self.size].to(device)
        queue_sim = z_T @ queue_tensor.T

        # Mask out near-identical embeddings (> 0.9999) to avoid retrieving self
        queue_sim = torch.where(
            queue_sim > 0.9999, torch.tensor(-float("inf"), device=device), queue_sim
        )

        combined_sim = torch.cat([queue_sim, in_batch_sim], dim=-1)
        combined_candidates = torch.cat([queue_tensor, z_T], dim=0)

        nn_indices = combined_sim.argmax(dim=-1)
        return combined_candidates[nn_indices]

    def enqueue(self, z_T: torch.Tensor) -> None:
        """Enqueue detached report embeddings into the circular FIFO buffer."""
        B, D = z_T.shape
        z_detached = F.normalize(z_T.detach().to(self.device), dim=-1)
        if self.queue is None:
            self.queue = torch.zeros(
                (self.capacity, D), dtype=z_detached.dtype, device=self.device
            )

        if B >= self.capacity:
            self.queue.copy_(z_detached[-self.capacity :])
            self.ptr = 0
            self.is_full = True
            return

        end = self.ptr + B
        if end <= self.capacity:
            self.queue[self.ptr : end] = z_detached
            self.ptr = end
            if self.ptr == self.capacity:
                self.ptr = 0
                self.is_full = True
        else:
            first = self.capacity - self.ptr
            self.queue[self.ptr : self.capacity] = z_detached[:first]
            second = B - first
            self.queue[0:second] = z_detached[first:]
            self.ptr = second
            self.is_full = True


def compute_declip_loss(
    *,
    z_I: torch.Tensor,
    z_I_aug: torch.Tensor,
    z_T: torch.Tensor,
    z_T_aug: torch.Tensor,
    z_T_nn: torch.Tensor,
    logit_scale: torch.Tensor,
    targets: torch.Tensor | None = None,
    mv_weight: float = 1.0,
    nn_weight: float = 0.5,
    image_ss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the multi-view, nearest-neighbour, and unimodal DeCLIP objective.

    L_total = L_MV + lambda_NN * L_NN + lambda_I * L_I

    Where:
    - L_MV: average InfoNCE across the four pairs (X, R), (X', R), (X, R'), (X', R')
    - L_NN: contrastive loss on nearest-neighbour positive pairs (X, R_NN), (X', R_NN)
    - L_I: SimCLR-style image contrastive loss between physical views X and X'
    """
    scale = logit_scale.exp()

    def _term(logits: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        if targets is not None:
            return soft_target_contrastive_loss(logits.float(), targets)
        return symmetric_contrastive_loss(logits)

    # 1. Multi-View Cross-Modal Supervision (4 pairs)
    l_11 = _term(scale * (z_I @ z_T.T), targets)
    l_21 = _term(scale * (z_I_aug @ z_T.T), targets)
    l_12 = _term(scale * (z_I @ z_T_aug.T), targets)
    l_22 = _term(scale * (z_I_aug @ z_T_aug.T), targets)
    l_mv = (l_11 + l_21 + l_12 + l_22) / 4.0

    # 2. Nearest-Neighbour Report Supervision (2 pairs: (X, R_NN) and (X', R_NN))
    l_nn_1 = _term(scale * (z_I @ z_T_nn.T), targets=None)
    l_nn_2 = _term(scale * (z_I_aug @ z_T_nn.T), targets=None)
    l_nn = (l_nn_1 + l_nn_2) / 2.0

    # 3. Unimodal Image Self-Supervision (SimCLR between view 1 and view 2)
    l_img_ss = _term(scale * (z_I @ z_I_aug.T), targets=None)

    total = mv_weight * l_mv + nn_weight * l_nn + image_ss_weight * l_img_ss

    breakdown = {
        "loss_mv": float(l_mv.detach()),
        "loss_nn": float(l_nn.detach()),
        "loss_img_ss": float(l_img_ss.detach()),
    }
    return total, breakdown
