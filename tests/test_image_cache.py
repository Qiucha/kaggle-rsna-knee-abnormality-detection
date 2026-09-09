from __future__ import annotations

import torch

from contrastive_learning.data import TensorMemoryCache


def test_tensor_memory_cache_is_bounded_and_least_recently_used() -> None:
    cache = TensorMemoryCache(capacity=2)
    cache.put("a", torch.tensor([1]))
    cache.put("b", torch.tensor([2]))
    assert torch.equal(cache.get("a"), torch.tensor([1]))

    cache.put("c", torch.tensor([3]))

    assert len(cache) == 2
    assert cache.get("b") is None
    assert torch.equal(cache.get("a"), torch.tensor([1]))
    assert torch.equal(cache.get("c"), torch.tensor([3]))
