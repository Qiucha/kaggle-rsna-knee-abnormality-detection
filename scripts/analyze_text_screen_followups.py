#!/usr/bin/env python
"""Read-only follow-ups to the English/medical text error review.

Two questions the review raised and left open (its "Data-quality observations" and
finding 5):

  mojibake   Is the ``horn<mangled>root`` artifact in the stored cohort text, or only
             in the rendered review packet?
  hubs       Is the retrieval-hub behavior seen in PubMedBERT's worst-twenty
             ``text_to_image`` pairs a general property of the run, or a tail artifact?

Both are answered from saved artifacts. Nothing here loads a checkpoint, runs a model,
touches the test split, or writes to the repository. Run either or both:

    .venv/bin/python scripts/analyze_text_screen_followups.py mojibake
    .venv/bin/python scripts/analyze_text_screen_followups.py hubs
    .venv/bin/python scripts/analyze_text_screen_followups.py all

Written 2026-09-11. The rankings files it reads are the frozen ones from the completed
``english-medical-text`` queue; re-running it cannot change them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from experiment_tools.text_hubs import RUNS, load_rankings, monte_carlo_null  # noqa: E402

MANIFEST = REPO / "data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl"
SOURCE_CSV = REPO / "data/train.csv"
PACKET = REPO / "digest/english-medical-text-error-review-packet.md"

# A UTF-8 byte sequence decoded as cp1252 leaves a leading U+00E2 followed by two more
# characters from the cp1252 high range. Match by codepoint, never by eyeballing a
# glyph: a terminal or an editor can re-mangle the display either way.
CP1252_HIGH = (
    "-ÿ€‚ƒ„…†‡ˆ‰Š‹"
    "ŒŽ‘’“”•–—˜™š›"
    "œžŸ"
)
MOJIBAKE = re.compile(f"â[{CP1252_HIGH}][{CP1252_HIGH}]")

# The same two strings as raw bytes, for the source-CSV check.
MANGLED_EN_DASH = bytes([0xC3, 0xA2, 0xE2, 0x82, 0xAC, 0xE2, 0x80, 0x9C])
MANGLED_APOSTROPHE = bytes([0xC3, 0xA2, 0xE2, 0x82, 0xAC, 0xE2, 0x84, 0xA2])
CLEAN_EN_DASH = bytes([0xE2, 0x80, 0x93])
CLEAN_APOSTROPHE = bytes([0xE2, 0x80, 0x99])

PAIR6_UID = "study-3bd56a7b1951d10f"

# Studies the review names as hubs. It abbreviates some UIDs by their leading digits
# and some by their trailing digits, so these are matched as substrings.
REVIEW_HUB_FRAGMENTS = ["REDACTED", "REDACTED", "REDACTED"]


def load_manifest() -> dict[str, dict]:
    records = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["study_uid"]] = record
    return records


def report_mojibake() -> None:
    print("=" * 78)
    print("MOJIBAKE ORIGIN")
    print("=" * 78)

    raw = SOURCE_CSV.read_bytes()
    raw.decode("utf-8")  # raises if the supplied file is not itself valid UTF-8
    print(f"\n{SOURCE_CSV.relative_to(REPO)} (supplied source reports) is valid UTF-8.")
    print(f"  mangled en dashes:   {raw.count(MANGLED_EN_DASH):>4}")
    print(f"  mangled apostrophes: {raw.count(MANGLED_APOSTROPHE):>4}")
    print(f"  clean en dashes:     {raw.count(CLEAN_EN_DASH):>4}")
    print(f"  clean apostrophes:   {raw.count(CLEAN_APOSTROPHE):>4}")
    print("  -> the damage is sporadic and per-report, already present in the supplied data.")

    records = load_manifest()
    hits = []
    sequences: Counter[str] = Counter()
    for uid, record in records.items():
        found = MOJIBAKE.findall(record.get("report") or "")
        if found:
            hits.append(record)
            sequences.update(found)

    print(f"\n{MANIFEST.relative_to(REPO)}: {len(hits)} of {len(records)} records affected.")
    print("  report_language:    ", Counter(r.get("report_language") for r in hits).most_common())
    print("  report_provenance:  ", Counter(r.get("report_provenance") for r in hits).most_common())
    print("  split:              ", Counter(r.get("split") for r in hits).most_common())
    identical = sum(1 for r in hits if r.get("report") == r.get("source_report"))
    print(f"  report == source_report in {identical}/{len(hits)} of them")
    print("  distinct sequences: ", [(repr(s), n) for s, n in sequences.most_common()])

    record = records[PAIR6_UID]
    text = record["report"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    repaired = MOJIBAKE.sub("–", text)
    print(f"\nPair-6 study {PAIR6_UID[-8:]}:")
    print(f"  stored report_sha256:       {record['report_sha256']}")
    print(f"  sha256 of the stored text:  {digest}  (match: {digest == record['report_sha256']})")
    print(f"  sha256 if the text were repaired: "
          f"{hashlib.sha256(repaired.encode('utf-8')).hexdigest()}")
    print("  -> repairing the characters changes the per-report hash, and therefore the")
    print("     cohort manifest SHA-256 that every completed run is pinned to. It is a")
    print("     cohort-contract change, not a typo fix. Not done here.")

    packet = PACKET.read_text(encoding="utf-8")
    print(f"\n{PACKET.relative_to(REPO)}: {len(MOJIBAKE.findall(packet))} mangled sequence(s) "
          f"alongside {packet.count(chr(0x2013))} clean en dashes")
    print("  -> the packet renderer reproduces the stored text faithfully; it is not the cause.")


def report_hubs() -> None:
    print("=" * 78)
    print("RETRIEVAL-HUB GENERALITY")
    print("=" * 78)
    print("\n`top_uid` in the saved rankings is the rank-1 retrieval for each query, so this")
    print("covers the whole 441-study validation pool, not just the worst twenty. It does")
    print("not recover each query's full 441-long ordering, which was never saved.\n")

    rankings = {label: load_rankings(path) for label, path in RUNS.items()}

    pool = None
    for label, rows in rankings.items():
        for direction in ("text_to_image", "image_to_text"):
            subset = [r for r in rows if r["direction"] == direction]
            here = {r["positive_uid"] for r in subset}
            if pool is None:
                pool = here
            elif here != pool:
                sys.exit(f"candidate pool differs in {label}/{direction}; runs are not comparable")
    print(f"All runs share an identical {len(pool)}-study candidate pool.\n")

    resolved = {}
    for fragment in REVIEW_HUB_FRAGMENTS:
        matches = [uid for uid in sorted(pool) if fragment in uid]
        if len(matches) != 1:
            sys.exit(f"review fragment {fragment} matches {len(matches)} studies")
        resolved[fragment] = matches[0]

    records = load_manifest()

    for direction in ("text_to_image", "image_to_text"):
        print("-" * 78)
        print(direction)
        print("-" * 78)
        counters = {}
        for label, rows in rankings.items():
            subset = [r for r in rows if r["direction"] == direction]
            counts = Counter(r["top_uid"] for r in subset)
            counters[label] = counts
            common = counts.most_common()
            hit_at_1 = sum(1 for r in subset if r["top_uid"] == r["positive_uid"])
            biggest, top_two = common[0][1], common[0][1] + common[1][1]
            print(f"\n{label}")
            print(f"  {len(subset)} queries, R@1 {hit_at_1}/{len(subset)} = "
                  f"{100 * hit_at_1 / len(subset):.4f}%")
            print(f"  distinct rank-1 targets: {len(counts)}")
            print(f"  biggest hub absorbs {biggest} queries ({100 * biggest / len(subset):.1f}%); "
                  f"top two absorb {top_two} ({100 * top_two / len(subset):.1f}%)")
            print("  top five: " + ", ".join(f"...{u[-8:]} x{n}" for u, n in common[:5]))
            for fragment, uid in resolved.items():
                order = [u for u, _ in common]
                place = order.index(uid) + 1 if uid in order else None
                print(f"    review-named {fragment}: rank-1 for {counts.get(uid, 0)} queries"
                      + (f", the #{place} hub of {len(order)}" if place else ", never rank-1"))

        tops = {label: {u for u, _ in counters[label].most_common(10)} for label in RUNS}
        labels = list(RUNS)
        print("\n  hub identity across runs (top-10 sets):")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                overlap = tops[labels[i]] & tops[labels[j]]
                print(f"    {labels[i]} vs {labels[j]}: {len(overlap)}/10 shared")
        shared = set.intersection(*tops.values())
        print(f"    shared by all three: {len(shared)}/10")
        print()

    print("-" * 78)
    print("Monte-Carlo null: 441 queries each picking a rank-1 target uniformly at random")
    print("-" * 78)
    for key, value in monte_carlo_null(len(pool), len(pool)).items():
        print(f"  {key}: {value}")

    print("\n" + "-" * 78)
    print("Is the worst-twenty tail more hub-dense than a random twenty?")
    print("-" * 78)
    print("The tail view is what the review saw. This asks whether it over-represents hubs.")
    rng = random.Random(999)
    for label, rows in rankings.items():
        subset = [r for r in rows if r["direction"] == "text_to_image"]
        worst = sorted(subset, key=lambda r: -r["expected_rank"])[:20]
        observed = len(Counter(r["top_uid"] for r in worst))
        sampled = sorted(len(Counter(r["top_uid"] for r in rng.sample(subset, 20)))
                         for _ in range(20000))
        p = sum(1 for v in sampled if v <= observed) / len(sampled)
        print(f"  {label:24s} worst-20 distinct targets = {observed}; "
              f"random-20 mean = {sum(sampled)/len(sampled):.2f}; "
              f"P(random at least as concentrated) = {p:.4f}")
    print("  -> directionally consistent but weak at n=20; not a settled effect.")

    print("\n" + "-" * 78)
    print("What the biggest text_to_image hubs are")
    print("-" * 78)
    for label, rows in rankings.items():
        counts = Counter(r["top_uid"] for r in rows if r["direction"] == "text_to_image")
        print(f"\n{label}")
        for uid, n in counts.most_common(3):
            record = records[uid]
            text = (record.get("report") or "").strip()
            print(f"  ...{uid[-8:]} x{n}  provenance={record.get('report_provenance')} "
                  f"chars={len(text)} laterality={record.get('laterality')}")
            print(f"     {text[:120]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["mojibake", "hubs", "all"], nargs="?", default="all")
    args = parser.parse_args()

    if args.check in ("mojibake", "all"):
        report_mojibake()
        print()
    if args.check in ("hubs", "all"):
        report_hubs()


if __name__ == "__main__":
    main()
