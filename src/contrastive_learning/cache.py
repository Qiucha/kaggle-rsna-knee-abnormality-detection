from __future__ import annotations

from collections import OrderedDict

import torch


class TensorMemoryCache:
    """A bounded least-recently-used cache for preprocessed study tensors."""

    def __init__(self, capacity: int = 0):
        if capacity < 0:
            raise ValueError("cache capacity cannot be negative")
        self.capacity = capacity
        self._values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str) -> torch.Tensor | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.capacity == 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
