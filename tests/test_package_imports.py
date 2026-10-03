"""What importing the package is allowed to cost.

``artifacts.py`` and ``paths.py`` are leaves over the standard library, so a
command script that needs only those must not pay for the training stack. The
package re-exports :class:`TrainConfig` lazily to keep that true; these tests
fail if an eager import creeps back into ``__init__.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

HEAVY = ("torch", "numpy", "pydicom", "PIL")


def _modules_after(statement: str) -> set[str]:
    """Import ``statement`` in a fresh interpreter and report the heavy modules loaded."""
    probe = (
        f"import sys; sys.path.insert(0, {SRC!r})\n"
        f"{statement}\n"
        f"print(' '.join(name for name in {HEAVY!r} if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return set(completed.stdout.split())


def test_importing_the_package_stays_off_the_training_stack() -> None:
    assert _modules_after("import contrastive_learning") == set()


def test_importing_a_leaf_stays_off_the_training_stack() -> None:
    assert _modules_after("import contrastive_learning.artifacts") == set()
    assert _modules_after("import contrastive_learning.paths") == set()


def test_the_lazy_re_export_still_resolves() -> None:
    from contrastive_learning import TrainConfig
    from contrastive_learning.config import TrainConfig as Direct

    assert TrainConfig is Direct


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    import contrastive_learning

    try:
        contrastive_learning.not_a_real_name
    except AttributeError as error:
        assert "not_a_real_name" in str(error)
    else:  # pragma: no cover - the lazy hook must not swallow typos
        raise AssertionError("expected AttributeError")
