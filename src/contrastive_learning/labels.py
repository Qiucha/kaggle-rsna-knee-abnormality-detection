"""Read LM-extracted finding labels and turn them into a per-study target vector.

The contrastive loss treats every off-diagonal pair as a negative, which is wrong
whenever two studies describe the same knee pathology: a torn ACL with an effusion
is not a negative for another torn ACL with an effusion. These labels give a cheap
proxy for "how alike are these two studies clinically", which
:func:`contrastive_learning.model.label_similarity_targets` mixes into the target.

Two decisions here are load-bearing and easy to get wrong:

* **The ``__verdict`` column is the signal, not the bare value column.** The value
  column holds only five distinct numbers per finding -- it is a lookup from
  verdict and confidence, not a calibrated probability. Averaging it would make
  UNK behave like 0.28, a number that sits nearer NO than YES for no good reason.
* **UNK means unobserved, not absent.** A report that never mentions synovitis
  (84% of them) says nothing about whether there is any. Scoring UNK as agreement
  would make most pairs look similar on most findings and flatten the matrix into
  a constant, which is a null result that looks like a real one. Every UNK is
  therefore masked out, and a pair is scored only on the findings both observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from contrastive_learning.data import read_csv_rows


# Verdicts that carry information. Anything else -- UNK, blank, an unexpected
# spelling -- is treated as unobserved rather than guessed at.
_POSITIVE = "YES"
_NEGATIVE = "NO"

_VERDICT_SUFFIX = "__verdict"
_CONFIDENCE_SUFFIX = "__conf"


@dataclass(frozen=True)
class LabelTable:
    """Per-study finding vectors, plus the finding names they are indexed by.

    ``vectors`` holds +1 for YES and -1 for NO; ``masks`` holds 1 wherever the
    verdict was one of those two and 0 for every unobserved finding. Keeping the
    sign and the mask separate is what lets a pair be scored on its co-observed
    findings alone.
    """

    findings: tuple[str, ...]
    vectors: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def unobserved(self) -> torch.Tensor:
        """The all-zero mask handed out for a study the CSV does not cover."""
        return torch.zeros(len(self.findings), dtype=torch.float32)

    def get(self, study_uid: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Vector and mask for one study, all-unobserved when it is not labelled.

        An unlabelled study is not an error: it contributes no affinity to any
        pair, so its target row falls back to the plain one-hot diagonal and it
        trains exactly as it would have under the unmodified loss.
        """
        zeros = self.unobserved
        return self.vectors.get(study_uid, zeros), self.masks.get(study_uid, zeros)

    def coverage(self, study_uids: list[str]) -> dict[str, float]:
        """How much of a split these labels actually cover, for the run record."""
        covered = [uid for uid in study_uids if uid in self.vectors]
        observed = [float(self.masks[uid].sum()) for uid in covered]
        return {
            "studies": len(study_uids),
            "labelled_studies": len(covered),
            "findings": len(self.findings),
            "mean_observed_findings": round(sum(observed) / len(observed), 3) if observed else 0.0,
        }


def finding_columns(fieldnames: list[str]) -> tuple[str, ...]:
    """Every finding the CSV carries a verdict for, in file order.

    Order only has to be self-consistent -- the similarity is a dot product over
    findings -- but taking it from the header keeps it stable across reads and
    avoids hard-coding a list that would silently ignore a thirteenth finding.
    """
    names = [
        name[: -len(_VERDICT_SUFFIX)] for name in fieldnames if name.endswith(_VERDICT_SUFFIX)
    ]
    if not names:
        raise ValueError(
            "No '<finding>__verdict' columns found; this does not look like an LM label CSV"
        )
    return tuple(names)


def read_lm_labels(path: Path, findings: list[str] | None = None) -> LabelTable:
    """Load an LM label CSV into signed vectors and co-observation masks.

    ``findings`` restricts the target to a subset of the CSV's findings, in the
    order given. Fewer findings means denser co-observation but a **coarser**
    similarity: n binary findings admit at most 2**n distinct clinical profiles, so
    a short list collapses many studies onto identical vectors and the target
    drifts from "semantically alike" toward plain label smoothing. Measure the tie
    rate before trusting a restricted set.

    Confidence columns are read but deliberately unused: weighting by a
    self-reported confidence that takes five values adds a tunable knob without
    adding information, and it can wait until the plain version is shown to work.
    """
    path = Path(path)
    rows = read_csv_rows(path, {"StudyInstanceUID"})
    if not rows:
        raise ValueError(f"No rows in LM label CSV: {path}")
    available = finding_columns(list(rows[0]))
    if findings is None:
        findings = available
    else:
        missing = [name for name in findings if name not in available]
        if missing:
            raise ValueError(
                f"Findings not present in {path}: {missing}. Available: {list(available)}"
            )
        if len(set(findings)) != len(findings):
            raise ValueError(f"Duplicate findings requested: {findings}")
        findings = tuple(findings)

    vectors: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for row in rows:
        uid = row["StudyInstanceUID"].strip()
        if not uid:
            continue
        if uid in vectors:
            raise ValueError(f"Duplicate StudyInstanceUID in {path}: {uid}")
        vector = torch.zeros(len(findings), dtype=torch.float32)
        mask = torch.zeros(len(findings), dtype=torch.float32)
        for index, finding in enumerate(findings):
            verdict = row.get(f"{finding}{_VERDICT_SUFFIX}", "").strip().upper()
            if verdict == _POSITIVE:
                vector[index], mask[index] = 1.0, 1.0
            elif verdict == _NEGATIVE:
                vector[index], mask[index] = -1.0, 1.0
        vectors[uid] = vector
        masks[uid] = mask
    return LabelTable(findings=findings, vectors=vectors, masks=masks)
