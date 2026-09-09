"""Image-report contrastive pretraining for knee MRI studies.

See ``README.md`` in this package for the module map and the design contracts each
module upholds. :class:`TrainConfig` is re-exported because it is the one object
callers assemble before invoking anything else.
"""

from contrastive_learning.config import TrainConfig

__all__ = ["TrainConfig"]
