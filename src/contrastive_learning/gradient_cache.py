"""One full-batch CLIP loss with exact chunk-wise gradient replay.

First encode every study/report without storing autograd activations. Differentiate
the full similarity matrix with respect to those embeddings and temperature, then
re-encode each chunk and apply its embedding gradient by the chain rule. Replay the
RNG as well: augmentation and dropout must reproduce the first pass. The optimizer
and scaler advance once per full comparison batch, never once per chunk.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .model import symmetric_contrastive_loss
from .preprocessing import augment_study_images


def _rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


@contextmanager
def _replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


def encode_chunk(model, batch, start, stop, device, *, tower, augment=None):
    """Transfer only one chunk's pixels/tokens. Full-batch pixels stay on the host."""
    if tower == "image":
        images = batch["images"][start:stop].to(device, non_blocking=True)
        if augment is not None:
            images = augment(images)
        if "valid" in batch:
            valid = batch["valid"][start:stop].to(device, non_blocking=True)
            return model.encode_images(images, valid)
        return model.encode_images(images)
    return model.encode_text(batch["input_ids"][start:stop].to(device, non_blocking=True),
                             batch["attention_mask"][start:stop].to(device, non_blocking=True))


def backward_cached_clip(model, batch, device, scaler, config, *, augment=augment_study_images):
    """Populate scaled parameter gradients for one batch; caller clips/steps once."""
    count = len(batch["images"])
    size = config.contrastive_microbatch_size
    if not 1 <= size <= count or count < 2:
        raise ValueError("Gradient replay needs >=2 paired studies and a valid chunk size")
    if batch["images"].device.type != "cpu":
        raise ValueError("Gradient replay keeps full-batch pixels on the CPU")
    chunks = [(start, min(start + size, count)) for start in range(0, count, size)]
    embeddings, states = {}, {}
    enabled_amp = config.amp and device.type == "cuda"
    for tower in ("image", "text"):
        parts, states[tower] = [], []
        for start, stop in chunks:
            states[tower].append(_rng_state(device))
            with torch.no_grad(), torch.autocast(device.type, enabled=enabled_amp):
                out = encode_chunk(model, batch, start, stop, device, tower=tower,
                                   augment=augment if tower == "image" else None)
            parts.append(out.detach().float())
        embeddings[tower] = torch.cat(parts).requires_grad_()
    # Small [batch, batch] graph; the two towers' large graphs are absent here.
    logits = model.logit_scale.exp() * embeddings["image"] @ embeddings["text"].T
    loss = symmetric_contrastive_loss(logits)
    scaler.scale(loss).backward()
    for tower in ("image", "text"):
        gradients = embeddings[tower].grad
        for (start, stop), state in zip(chunks, states[tower], strict=True):
            with _replay_rng(state, device):
                with torch.autocast(device.type, enabled=enabled_amp):
                    out = encode_chunk(model, batch, start, stop, device, tower=tower,
                                       augment=augment if tower == "image" else None)
                # These embedding gradients already include GradScaler's scale.
                torch.autograd.backward(out.float(), gradients[start:stop])
    return loss.detach()
