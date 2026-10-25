"""Confirm or refute the acquisition screen's candidate duplicate groups by pixels.

The screen in `audit_acquisition_duplicates.py` matches on headers, which is strong
evidence of one acquisition but not proof of identical images. This decodes the
selected series and compares an ordered decoded-pixel hash per plane -- the same
evidence the known quarantined pair was verified with.

Any candidate group containing a test-split study is SKIPPED and reported as gated.
Decoding a locked-test volume is a protocol decision for the user, per the feedback
response's section 8; this script does not take it.

Output: digest/duplicate-group-pixel-verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from contrastive_learning.manifest import read_manifest_rows
from experiment_tools.dicom_series import plane_digest

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
SCREEN = ROOT / "digest/acquisition-duplicate-screen.json"
OUT = ROOT / "digest/duplicate-group-pixel-verification.json"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = {row["study_uid"]: row for row in read_manifest_rows(COHORT)}
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))

    results, gated = [], []
    started = time.time()
    for group in screen["stage2"]["detail"]:
        if group["involves_test"]:
            gated.append(
                {
                    "members": group["members"],
                    "splits": group["splits"],
                    "study_uids": group["study_uids"],
                    "reason": (
                        "contains a locked-test study; decoding its pixels is a protocol "
                        "decision for the user, not taken here"
                    ),
                }
            )
            continue
        per_study = {}
        for uid in group["study_uids"]:
            row = rows[uid]
            per_study[uid] = {
                "split": row["split"],
                "training_eligible": row["training_eligible"],
                "report_sha256": row["report_sha256"],
                "source_report_sha256": row["source_report_sha256"],
                "laterality": row["laterality"],
                "planes": {},
            }
            for plane, directory in sorted(row["slots"].items()):
                digest = plane_digest(directory)
                per_study[uid]["planes"][plane] = {
                    "ordered_decoded_pixel_sha256": digest["ordered_decoded_pixel_sha256"],
                    "slices": digest["slices"],
                }

        planes = sorted({p for s in per_study.values() for p in s["planes"]})
        identical_planes = {
            plane: len({s["planes"][plane]["ordered_decoded_pixel_sha256"] for s in per_study.values()}) == 1
            for plane in planes
        }
        reports = {s["report_sha256"] for s in per_study.values()}
        results.append(
            {
                "members": group["members"],
                "splits": group["splits"],
                "crosses_splits": group["crosses_splits"],
                "pixels_identical_on_every_plane": all(identical_planes.values()),
                "identical_planes": identical_planes,
                "distinct_translated_reports": len(reports),
                "reports_conflict": len(reports) > 1,
                "all_members_training_eligible": all(s["training_eligible"] for s in per_study.values()),
                "study_uids": group["study_uids"],
                "per_study": per_study,
            }
        )

    confirmed = [r for r in results if r["pixels_identical_on_every_plane"]]
    audit = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_screen": str(SCREEN.relative_to(ROOT)),
        "method": (
            "sha256 over every decoded slice array of each selected series, in "
            "`ordered_slice_paths` order -- the pipeline's own geometric ordering, not "
            "file name order."
        ),
        "groups_examined": len(results),
        "groups_gated_on_locked_test": len(gated),
        "gated": gated,
        "groups_confirmed_identical": len(confirmed),
        "studies_in_confirmed_groups": sum(r["members"] for r in confirmed),
        "confirmed_groups_crossing_splits": sum(1 for r in confirmed if r["crosses_splits"]),
        "confirmed_groups_with_conflicting_reports": sum(1 for r in confirmed if r["reports_conflict"]),
        "confirmed_studies_still_training_eligible": sum(
            sum(1 for s in r["per_study"].values() if s["training_eligible"]) for r in confirmed
        ),
        "seconds": round(time.time() - started, 1),
        "detail": results,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "detail"}, indent=1))
    for r in results:
        print(
            r["members"], r["splits"], "identical" if r["pixels_identical_on_every_plane"] else "DIFFERENT",
            "reports_conflict" if r["reports_conflict"] else "reports_agree",
            r["identical_planes"],
        )


if __name__ == "__main__":
    main()
