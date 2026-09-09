"""Durable, self-describing artifact writes and content hashes.

Every experiment artifact this project produces -- run manifests, metrics
histories, cohort manifests, evaluation reports, cached tensors -- is written
through this module so that two properties hold everywhere without each caller
restating them:

* A reader never observes a partial file. Writes land in a sibling temporary
  file and are renamed into place, which is atomic within a directory.
* Identity is a SHA-256 of file bytes. Protocols freeze these hashes, so the
  digest must be computed the same way for every artifact.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    """Hash a file's bytes in chunks, so large checkpoints do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Hash decoded text, for report bodies compared across CSV encodings."""
    return hashlib.sha256(value.encode()).hexdigest()


def write_atomic(path: Path, write: Callable[[Path], None]) -> Path:
    """Run ``write`` against a temporary sibling, then rename it onto ``path``.

    The temporary name is unique, so concurrent writers of the same artifact
    cannot corrupt each other's payloads; the loser of the rename race simply
    has its bytes replaced by an equivalent file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_text(path: Path, text: str) -> Path:
    return write_atomic(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def write_json(path: Path, value: Any) -> Path:
    """Write one pretty-printed, key-sorted JSON document for stable diffs."""
    return write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, ensure_ascii: bool = True) -> Path:
    """Write one compact, key-sorted JSON document per line."""
    return write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=ensure_ascii) + "\n" for row in rows
        ),
    )


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    """Read a JSON-lines artifact, ignoring the blank line a writer may append."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]
