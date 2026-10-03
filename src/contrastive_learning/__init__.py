"""Image-report contrastive pretraining for knee MRI studies.

See ``README.md`` in this package for the module map and the design contracts each
module upholds. :class:`TrainConfig` is re-exported because it is the one object
callers assemble before invoking anything else.

The re-export is resolved lazily (PEP 562), so importing this package costs
nothing beyond the standard library. ``artifacts.py`` and ``paths.py`` are leaves
over the standard library by design, and a command script that needs only those
must not pay for torch, numpy, pydicom and PIL by way of ``TrainConfig``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contrastive_learning.config import TrainConfig

__all__ = ["TrainConfig"]


def __getattr__(name: str) -> object:
    """Resolve the re-export on first use, leaving submodule imports untouched."""
    if name == "TrainConfig":
        from contrastive_learning.config import TrainConfig

        return TrainConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the re-export alongside whichever submodules have been imported."""
    names = {*__all__, *(name for name in globals() if not name.startswith("_"))}
    return sorted(names - {"TYPE_CHECKING"})
