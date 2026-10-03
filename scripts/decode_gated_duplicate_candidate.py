"""Settle the one train/test duplicate candidate the screen left gated.

Run under an explicit user decision, 2026-09-11. The acquisition screen flagged a train
study and a test study as the same acquisition; only decoded pixels can confirm or clear
it, and the test member's pixels were left alone until the user chose.

**Locked-test discipline, deliberately narrow.** This reads the test study's DICOM pixels
and nothing else about it. It does not read its report text, its labels, or any metric;
it does not load a model or score anything. Reports are compared by sha256 from the
manifest, which answers "do these two reports agree?" without exposing either one. The
output is an integrity verdict, and no information that could inform model selection
crosses the boundary.

Output: digest/gated-duplicate-resolution.json
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from contrastive_learning.manifest import read_manifest_rows
from contrastive_learning.preprocessing import _decode_pixels, ordered_slice_paths

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
SCREEN = ROOT / "digest/acquisition-duplicate-screen.json"
OUT = ROOT / "digest/gated-duplicate-resolution.json"


def plane_digest(directory: str) -> dict:
    """Ordered decoded-pixel hash for one series, plus shape facts for the record."""
    digest = hashlib.sha256()
    paths = ordered_slice_paths(Path(directory))
    shapes, dtypes = set(), set()
    for path in paths:
        pixels = _decode_pixels(path)
        if pixels is None:
            digest.update(b"<undecodable>")
            continue
        array = np.ascontiguousarray(pixels)
        shapes.add(tuple(array.shape))
        dtypes.add(str(array.dtype))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {
        "ordered_decoded_pixel_sha256": digest.hexdigest(),
        "slices": len(paths),
        "slice_shapes": sorted(str(s) for s in shapes),
        "dtypes": sorted(dtypes),
    }


def main() -> None:
    rows = {row["study_uid"]: row for row in read_manifest_rows(COHORT)}
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    gated = [g for g in screen["stage2"]["detail"] if g["involves_test"]]
    if len(gated) != 1:
        raise SystemExit(f"expected exactly one gated group, found {len(gated)}")
    uids = gated[0]["study_uids"]

    started = time.time()
    per_study = {}
    for uid in uids:
        row = rows[uid]
        per_study[uid] = {
            "split": row["split"],
            "training_eligible": row["training_eligible"],
            "laterality": row["laterality"],
            # Hashes only. The text itself is never read for either study, so the
            # comparison is valid without the test report leaving its split.
            "report_sha256": row["report_sha256"],
            "source_report_sha256": row["source_report_sha256"],
            "planes": {
                plane: plane_digest(directory) for plane, directory in sorted(row["slots"].items())
            },
        }

    planes = sorted({p for s in per_study.values() for p in s["planes"]})
    identical = {
        plane: len({s["planes"][plane]["ordered_decoded_pixel_sha256"] for s in per_study.values()}) == 1
        for plane in planes
    }
    all_identical = all(identical.values())
    reports_agree = len({s["report_sha256"] for s in per_study.values()}) == 1
    sources_agree = len({s["source_report_sha256"] for s in per_study.values()}) == 1
    test_uid = next(u for u in uids if per_study[u]["split"] == "test")
    train_uid = next(u for u in uids if per_study[u]["split"] == "train")

    if all_identical:
        verdict = (
            "CONFIRMED DUPLICATE: a test study and a training study are the same images. "
            "The test split is contaminated for this study -- it cannot measure "
            "generalisation for an acquisition the model trained on."
        )
    elif any(identical.values()):
        verdict = (
            "PARTIAL: some planes match and others do not, which no clean duplication "
            "explains. Treat as a join or export anomaly and investigate before relying "
            "on either study."
        )
    else:
        verdict = (
            "CLEARED: same acquisition protocol and slice geometry, different images. "
            "Not a duplicate; the test split is unaffected."
        )

    audit = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "authorisation": (
            "User instruction, 2026-09-11, to decode the gated train/test candidate. This "
            "is the locked-test protocol decision the feedback response's section 8 "
            "reserves for the user."
        ),
        "scope": {
            "reads_test_pixels": True,
            "reads_test_report_text": False,
            "reads_test_labels": False,
            "loads_a_model_or_computes_a_metric": False,
            "note": (
                "Reports are compared by manifest sha256, so agreement is established "
                "without either report being read."
            ),
        },
        "train_study": train_uid,
        "test_study": test_uid,
        "pixels_identical_on_every_plane": all_identical,
        "identical_planes": identical,
        "reports_identical": reports_agree,
        "source_reports_identical": sources_agree,
        "verdict": verdict,
        "seconds": round(time.time() - started, 1),
        "per_study": per_study,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "per_study"}, indent=1))
    for uid, data in per_study.items():
        print(uid[-16:], data["split"], {p: d["ordered_decoded_pixel_sha256"][:16] for p, d in data["planes"].items()})


if __name__ == "__main__":
    main()
