"""Machine-readable catalog of every file in ``digest/`` (depth 1).

Each entry says which site page owns the file (``page``), which other pages
also list it (``also_on``), what kind of artefact it is, its status, and a
one-line purpose. ``check_complete`` makes the build fail when a file appears
in ``digest/`` without a catalog row, so the "every file has a home" invariant
survives future additions: add a row here when you add a digest file.

Status vocabulary (exactly one per file):
    current      the live record for its question
    superseded   replaced by a later file (named in ``caveat``)
    amended      edited after publication; read the in-page amendment
    retracted    a key conclusion was withdrawn
    not-adopted  the change it tested was not adopted
    null         the experiment returned a null result
    historical   an early, still-true record that no longer drives decisions

``caveat`` carries what the status enum cannot (single seed, exposed cohort,
AMBIGUOUS verdict, privacy). It is rendered as a secondary badge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .sitemap import BY_SLUG

STATUSES = ("current", "superseded", "amended", "retracted", "not-adopted", "null", "historical", "excluded")

KINDS = {
    "result": "Result write-up",
    "protocol": "Protocol / pre-registration",
    "audit": "Audit",
    "gate": "Decision gate",
    "review": "Review",
    "research": "Research notes",
    "narrative": "Narrative",
    "ops": "Operations",
    "index": "Index",
    "raw-data": "Raw data",
    "probe": "Probe run (JSON)",
    "paired": "Paired comparison (JSON)",
    "features": "Cached features",
    "figure": "Figure",
    "report-html": "Legacy HTML report",
    "directory": "Directory",
}


@dataclass(frozen=True)
class Entry:
    path: str  # relative to digest/
    page: str
    also_on: tuple[str, ...]
    kind: str
    status: str
    purpose: str
    caveat: str = ""
    original_report: str = ""  # legacy root HTML that renders this file (relative to digest/)

    @property
    def name(self) -> str:
        return self.path.rstrip("/")

    @property
    def is_markdown(self) -> bool:
        return self.path.endswith(".md")

    @property
    def is_legacy_html(self) -> bool:
        return self.path.endswith(".html") and self.path != "index.html"

    @property
    def pages(self) -> tuple[str, ...]:
        return (self.page, *self.also_on)


def E(path, page, also, kind, status, purpose, caveat="", original=""):
    also_on = tuple(a for a in also.split(",") if a)
    return Entry(path, page, also_on, kind, status, purpose, caveat, original)


MRI = "Embeds a real MRI preview: local only, do not host publicly"

ENTRIES: list[Entry] = [
    # --- site plumbing -------------------------------------------------------
    E("index.html", "library", "", "index", "current",
      "Front door: redirects to the navigable site."),
    E("handoff/", "library", "", "directory", "excluded",
      "Local working notes with machine state (PIDs, paths); not published.",
      "Gitignored: never linked from the site"),
    E("README.md", "library", "", "index", "current",
      "Digest index and the house conventions for quoting numbers."),
    E("HOSTING.md", "library", "", "ops", "current",
      "How to serve the digest and the site locally."),
    # --- pipeline ------------------------------------------------------------
    E("architecture-parameter-audit.json", "pipeline", "", "audit", "historical",
      "Trainable vs frozen parameter counts, baseline vs one image block.",
      "Pre-LoRA arms only"),
    E("cleanup-reproduction-20260908.json", "pipeline", "", "raw-data", "historical",
      "Per-metric deltas and hashes for the cleanup reproduction run."),
    E("cleanup-reproduction-20260908.md", "pipeline", "", "audit", "historical",
      "Does the cleaned code reproduce the frozen baseline bit-for-bit?"),
    E("contrastive-checkpoint-rescore.json", "pipeline", "", "audit", "historical",
      "Did the tie-aware retrieval fix change any stored pilot checkpoint score?"),
    E("contrastive-learning-current-state.md", "pipeline", "", "audit", "superseded",
      "What worked and what was broken, as of 2026-09-04.",
      "Pre-full-cohort snapshot; its P0 items were fixed (implementation-status)"),
    E("contrastive-learning.html", "pipeline", "", "report-html", "amended",
      "Hand-written project narrative: what moved the number, what is open.",
      "Hand-maintained; omits LoRA and the Kaggle submission"),
    E("contrastive-learning-implementation-status.md", "pipeline", "", "narrative", "historical",
      "Which audit items were fixed, and how the GPU matrix was prepared.",
      "Amended in place 2026-09-11 (patient independence unverifiable)"),
    E("contrastive-learning-pipeline.md", "pipeline", "", "protocol", "superseded",
      "End-to-end design reference: data, preprocessing, towers, loss.",
      "Model choices superseded; preprocessing sections still valid"),
    E("experiment-session-log.md", "pipeline", "", "result", "historical",
      "58-study CPU pilot: which text encoder and settings retrieved best?",
      "12-study folds; pilot only"),
    E("full-manifest-migration.json", "pipeline", "", "audit", "current",
      "Did the manifest migration preserve bytes, rows and splits?"),
    E("full-projection-v1-curves.html", "pipeline", "", "report-html", "historical",
      "Per-epoch curves of the frozen-tower projection baseline."),
    E("full-projection-v1-handoff.md", "pipeline", "", "ops", "superseded",
      "How the baseline was blocked, unblocked on CUDA, and run.",
      "Self-labelled historical handoff"),
    E("full-projection-v1-results.md", "pipeline", "", "result", "historical",
      "Does frozen-tower projection training learn full-cohort retrieval?",
      original="full-projection-v1-curves.html"),
    E("image-shape-audit.md", "pipeline", "data", "audit", "amended",
      "Tensor shapes from DICOM slices to the study embedding.",
      "Correction 2026-09-10: the trace harness no longer exists"),
    E("image-shape-trace.jsonl", "pipeline", "", "raw-data", "historical",
      "Evidence trace behind the tensor-shape table.",
      "Not regenerable"),
    E("lora-memory-profile-bs16.json", "pipeline", "adaptation", "raw-data", "current",
      "Peak GPU memory by depth and rank at batch 16 (to 12 blocks)."),
    E("lora-memory-profile.json", "pipeline", "adaptation", "raw-data", "current",
      "Which full-FT / LoRA depth and rank configs fit 8 GB at batch 32?"),
    E("oom-solutions-benchmark.json", "pipeline", "adaptation", "raw-data", "current",
      "Which memory tricks let deeper adaptation fit in 8 GB?"),
    E("project-state.html", "pipeline", "", "report-html", "amended",
      "Generated run-status snapshot with architecture explorer and evidence cards.",
      "Hero and architecture panel predate LoRA; text-block card is stale"),
    E("translated-report-training.html", "pipeline", "", "report-html", "historical",
      "Pilot page: which medical text encoder aligns translated reports with MRI?",
      MRI),
    # --- data ----------------------------------------------------------------
    E("acquisition-duplicate-screen.json", "data", "", "audit", "current",
      "Do any studies share an identical acquisition header?",
      "Its test-involving group was later cleared"),
    E("acquisition-metadata-trust.md", "data", "overfitting", "audit", "current",
      "What the series CSV hides: flags, FOV, depth, slots, laterality, leak.",
      "Its 150 mm crop pick was later not adopted"),
    E("acquisition-slots.json", "data", "", "audit", "current",
      "Which weighting / plane / fat-sat slots exist, and which are picked."),
    E("crop-retention.json", "data", "overfitting", "audit", "current",
      "How much tissue each centred crop size keeps.",
      "The 150 mm crop it motivated was not adopted"),
    E("duplicate-group-pixel-verification.json", "data", "", "audit", "amended",
      "Are the screened duplicate groups pixel-identical?",
      "Amended 2026-09-11: the gated group was cleared"),
    E("fov-and-depth-audit.json", "data", "", "audit", "current",
      "True field of view, and how much depth three slices cover."),
    E("full-projection-v1-data-audit.json", "data", "", "raw-data", "historical",
      "Machine counts behind the baseline data review."),
    E("full-projection-v1-data-review.md", "data", "", "audit", "superseded",
      "Do duplicate reports or truncation undermine the frozen baseline?",
      "Superseded by integrity-coverage-audit.md"),
    E("gated-duplicate-resolution.json", "data", "", "audit", "current",
      "Decision record: is the train/test candidate pair a duplicate?"),
    E("gold-case-inventory.json", "data", "probes", "audit", "current",
      "Provenance and training exposure of each gold study."),
    E("gold-case-inventory.md", "data", "probes", "audit", "current",
      "What the 58 gold studies are, and what they can support.",
      "Its probe contract later superseded in practice by the gold holdout"),
    E("integrity-coverage-audit.json", "data", "", "audit", "current",
      "Machine counts for cross-split duplicates, joins and patient leakage."),
    E("integrity-coverage-audit.md", "data", "", "audit", "current",
      "Are there cross-split duplicates, broken joins, or patient leakage?",
      "Study-disjoint only; patient independence is unverifiable"),
    E("multilingual-text-qc.json", "data", "", "audit", "current",
      "Does E5 tokenize every report, and how many exceed 512 tokens?"),
    E("protocol-leak-check.json", "data", "probes", "gate", "current",
      "Does the series count leak gold labels?"),
    E("report-length-statistics.json", "data", "", "audit", "current",
      "How often the 512-token limit truncates reports and their conclusions."),
    # --- text ----------------------------------------------------------------
    E("claude-translated-9bec1bd-data-audit.json", "text", "", "audit", "current",
      "Identity, duplicate and leakage checks on the 9bec1bd cohort."),
    E("claude-translated-baseline-audit.json", "text", "", "audit", "current",
      "Was the 9bec1bd cohort derived correctly, with only text changed?"),
    E("claude-translated-baseline.md", "text", "", "result", "amended",
      "Does the first Claude-translated corpus (9bec1bd) train a comparable baseline?",
      "Launch notes marked historical; see translation-update-dfd07db.md"),
    E("claude-translated-dfd07db-baseline-audit.json", "text", "", "audit", "current",
      "Derivation checks for the dfd07db translated cohort."),
    E("claude-translated-dfd07db-comparison-audit.json", "text", "", "audit", "current",
      "Machine-verified numbers for the impression-first ordering comparison."),
    E("claude-translated-dfd07db-comparison-results.md", "text", "", "result", "current",
      "Does impression-first ordering beat full report text?",
      "Single seed (2026); decision: keep full text"),
    E("claude-translated-dfd07db-continuation.md", "text", "", "review", "amended",
      "Input audit, worst-20 review, quarantine, and the ordering protocol."),
    E("claude-translated-dfd07db-duplicate-verification.json", "text", "", "audit", "current",
      "Are the two flagged studies truly identical images?"),
    E("claude-translated-dfd07db-image-block-audit.json", "text", "adaptation", "audit", "current",
      "Verification of the image-block runs and their paired differences."),
    E("claude-translated-dfd07db-image-block-error-review.json", "text", "adaptation", "raw-data", "current",
      "Subgroup slices and worst-20 pairs for the image-block runs."),
    E("claude-translated-dfd07db-image-block-error-review.md", "text", "adaptation", "review", "current",
      "Where image adaptation gains and loses; the worst 20 pairs."),
    E("claude-translated-dfd07db-image-block-protocol.md", "text", "adaptation", "protocol", "current",
      "Frozen protocol: unfreeze the final DINOv2 block, three-seed repeat rule."),
    E("claude-translated-dfd07db-image-block-results.md", "text", "adaptation", "result", "amended",
      "Does adapting the final image block beat the frozen baseline?"),
    E("claude-translated-dfd07db-input-audit.json", "text", "", "audit", "current",
      "Truncation, lost impression headings and duplicate images in train+val."),
    E("claude-translated-dfd07db-ordering-audit.json", "text", "", "audit", "current",
      "Did impression-first reordering keep every word and move late summaries?"),
    E("claude-translated-dfd07db-results.md", "text", "", "result", "current",
      "Full-cohort dfd07db translated anchor retrieval score."),
    E("claude-translated-dfd07db-text-block-protocol.md", "text", "", "protocol", "superseded",
      "Frozen protocol for adapting the final text block, three seeds.",
      "Says 'not launched', but the runs happened (validation-only; did not help)"),
    E("config-translation-audit.json", "text", "", "audit", "historical",
      "Mechanical counts across the three translation columns of the teammate CSV."),
    E("config-translation-review.md", "text", "", "review", "historical",
      "Is the teammate's Google-translated corpus fit for training?"),
    E("english-medical-text-error-review.md", "text", "", "review", "amended",
      "Why the two English text-encoder swaps failed, and what next.",
      "Follow-up 2026-09-11 overturns finding 5 (hubs are universal)"),
    E("english-medical-text-preflight.json", "text", "", "audit", "current",
      "Per-tokenizer length and truncation profile before training."),
    E("english-medical-text-protocol.md", "text", "", "protocol", "current",
      "Frozen protocol: swap only the frozen text encoder, seed 2026."),
    E("english-medical-text-results.json", "text", "", "raw-data", "current",
      "Full metrics, subgroups and worst pairs for both text-encoder screens."),
    E("english-medical-text-results.md", "text", "", "result", "current",
      "Do frozen e5-base-v2 or PubMedBERT beat frozen multilingual E5?",
      "Single seed (2026)"),
    E("translation-update-dfd07db-audit.json", "text", "", "audit", "current",
      "Row-level diff of the translation CSVs, 9bec1bd to dfd07db."),
    E("translation-update-dfd07db.md", "text", "", "review", "current",
      "Do five manual translation edits require restarting the baseline?"),
    # --- adaptation ----------------------------------------------------------
    E("both-block-depth-features.npz", "adaptation", "probes", "features", "current",
      "Cached probe features for the both-tower depth and LoRA models."),
    E("both-block-depth-results.md", "adaptation", "", "result", "current",
      "How many blocks per tower, and which LoRA rank and batch, work best?",
      "Single seed (2026); its 'record' / '0% inflation' wording is not endorsed",
      original="both-block-depth-v1.html"),
    E("both-block-depth-v1.html", "adaptation", "", "report-html", "current",
      "Interactive legacy page: both-tower depth, LoRA, memory, holdout.",
      "Single seed; its 'record' and '0% inflation' claims are not endorsed"),
    E("both-tower-adaptation-summary.md", "adaptation", "", "result", "current",
      "Does adapting the text block help the image readout? Which depth wins?",
      "Development readouts on exposed gold"),
    E("image-block-variants-curves.html", "adaptation", "", "report-html", "current",
      "Loss and R@1 curves for the fifteen image-block variant runs."),
    E("image-block-variants-v1.md", "adaptation", "", "result", "current",
      "Which lever moves label readout: language, depth, or schedule?",
      "Levels are development statistics (exposed gold)",
      original="image-block-variants-curves.html"),
    E("lora-adaptation-summary.md", "adaptation", "", "result", "current",
      "How LoRA ranks compare with full fine-tuning at six blocks; holdout level.",
      "Exposed and holdout rows are not a paired generalisation estimate"),
    E("lora-holdout-features.npz", "adaptation", "", "features", "current",
      "Cached probe features for the gold-holdout LoRA encoders."),
    E("plane-pooling-analysis.json", "adaptation", "", "audit", "current",
      "Why learned plane attention reads labels worse than a uniform mean.",
      "Built on the older one-block checkpoints"),
    E("plane-pooling-comparisons-v1.json", "adaptation", "probes", "paired", "null",
      "Seed-matched uniform pooling vs learned attention, paired probe contrast."),
    E("plane-pooling-features.npz", "adaptation", "", "features", "current",
      "Cached probe features for the uniform-pooling arm."),
    E("plane-pooling-protocol.md", "adaptation", "", "protocol", "amended",
      "Pre-registration: will uniform plane pooling beat learned attention?",
      "Addendum corrects a mis-attributed number and the plane order"),
    E("plane-pooling-v1.md", "adaptation", "", "result", "null",
      "Does training with uniform plane pooling improve the gold probe?"),
    # --- overfitting ---------------------------------------------------------
    E("budget-40ep-v1.json", "overfitting", "", "raw-data", "current",
      "Per-seed 40-epoch paired R@1 for drop-path 0.1 vs none.",
      "Verdict AMBIGUOUS"),
    E("budget-40ep-v1.md", "overfitting", "", "result", "current",
      "Does the drop-path 0.1 deficit persist when both arms train 40 epochs?",
      "Pre-registered verdict AMBIGUOUS: do not round"),
    E("convergence-audit.json", "overfitting", "", "audit", "current",
      "Did the drop-path and crop150 arms converge within 20 epochs?"),
    E("convergence-curves.html", "overfitting", "", "report-html", "current",
      "Were the regularised arms, and crop150, truncated by the budget?"),
    E("crop150-memorisation.json", "overfitting", "", "raw-data", "not-adopted",
      "Matched-N memorisation gap, crop130 vs crop150."),
    E("crop150-probe-v1.md", "overfitting", "", "result", "not-adopted",
      "Does the 150 mm crop help the twelve gold labels?"),
    E("crop150-rebaseline-v1.md", "overfitting", "", "result", "not-adopted",
      "Does a 150 mm crop (more tissue retained) change retrieval?"),
    E("crop150-vs-crop130-comparisons-v1.json", "overfitting", "probes", "paired", "not-adopted",
      "Seed-matched probe contrast, crop150 vs crop130."),
    E("drop-path-5e-2-arm.json", "overfitting", "", "raw-data", "superseded",
      "Per-seed detail of the drop-path 0.05 arm.",
      "Superseded in scope by the drop-path sweep"),
    E("drop-path-5e-2-arm.md", "overfitting", "", "result", "superseded",
      "Does stochastic depth 0.05 reduce memorisation and keep retrieval?",
      "Superseded in scope, confirmed in substance by the sweep (2026-09-15)"),
    E("drop-path-sweep-v1.json", "overfitting", "", "raw-data", "amended",
      "Drop-path dose-response per rate and seed.",
      "See the 2026-09-15 amendment in the .md"),
    E("drop-path-sweep-v1.md", "overfitting", "", "result", "amended",
      "Is there a drop-path rate that cuts memorisation without costing retrieval?",
      "Amended 2026-09-15: rates confounded with the 20-epoch budget"),
    E("duplicate-confusion-audit.json", "overfitting", "", "audit", "current",
      "Per-miss duplicate-report counts behind the confusion audit."),
    E("duplicate-confusion-audit.md", "overfitting", "", "gate", "current",
      "Do duplicate reports explain retrieval R@1 misses?",
      "Amends the false-negative audit's detector finding"),
    E("false-negative-rate-audit.json", "overfitting", "", "gate", "amended",
      "False-negative pair counts per batch at batch size 32.",
      "Detector finding void; see the .md amendment"),
    E("false-negative-rate-audit.md", "overfitting", "", "gate", "amended",
      "Are false-negative pairs common enough at batch 32 to need soft targets?",
      "Amended 2026-09-13: rate finding stands, detector finding void"),
    E("hard-negative-gate-data.json", "overfitting", "", "gate", "current",
      "Does the top-1 negative already dominate the InfoNCE denominator?",
      "Computed, not a decision: the arm is not authorised"),
    E("matched-n-loss-audit.json", "overfitting", "", "raw-data", "current",
      "Per-seed matched-candidate train vs validation loss."),
    E("matched-n-loss-audit.md", "overfitting", "", "audit", "current",
      "Does the four-block image tower overfit at a matched candidate count?",
      "Replaces the withdrawn val-loss inference"),
    E("val-loss-rank-depth.json", "overfitting", "", "audit", "amended",
      "Do scale-free ranks improve over the late training window?",
      "Companion to the amended val-loss audit (conclusion 1 withdrawn)"),
    E("val-loss-vs-retrieval-audit.json", "overfitting", "", "gate", "amended",
      "Machine numbers behind the val-loss vs R@1 audit.",
      "AMENDED 2026-09-13: conclusion 1 withdrawn — the model overfits"),
    E("val-loss-vs-retrieval-audit.md", "overfitting", "", "gate", "amended",
      "Why do validation loss and validation R@1 disagree late in training?",
      "AMENDED 2026-09-13: conclusion 1 withdrawn — the model overfits"),
    # --- probes --------------------------------------------------------------
    E("downstream-both-block-v1.md", "probes", "", "result", "current",
      "Does adapting the text block too add medical semantics to the image?"),
    E("downstream-both-vs-image-block-planes-seed2026.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, plane-attention feature, seed 2026."),
    E("downstream-both-vs-image-block-planes-seed2027.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, plane-attention feature, seed 2027."),
    E("downstream-both-vs-image-block-planes-seed2028.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, plane-attention feature, seed 2028."),
    E("downstream-both-vs-image-block-seed2026.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, uniform-mean feature, seed 2026."),
    E("downstream-both-vs-image-block-seed2027.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, uniform-mean feature, seed 2027."),
    E("downstream-both-vs-image-block-seed2028.json", "probes", "adaptation", "paired", "current",
      "Both-block minus image-block, uniform-mean feature, seed 2028."),
    E("downstream-extended-comparisons-v1.json", "probes", "adaptation", "paired", "current",
      "Paired contrasts: translated vs original, 40 vs 20 epochs, depth steps.",
      "v1/v2/v3 hold different contrasts, not a supersession chain"),
    E("downstream-extended-comparisons-v2.json", "probes", "adaptation", "paired", "current",
      "Paired contrasts: four vs one block at more seeds, eight blocks, shuffled control."),
    E("downstream-extended-comparisons-v3.json", "probes", "adaptation", "paired", "current",
      "Shuffled-pairing control: four blocks minus shuffled, three seeds."),
    E("downstream-original-vs-translated-v1.json", "probes", "text", "paired", "current",
      "Translated minus original reports, seed 2026, both pooling features."),
    E("downstream-original-vs-translated-v1.md", "probes", "text", "result", "amended",
      "Does translating reports change what the image tower learns?",
      "Partly superseded by its own update and image-block-variants-v1.md"),
    E("downstream-probe-both-block-depth-v1.json", "probes", "adaptation", "probe", "current",
      "Gold probe: both-tower depth and LoRA depth/rank sweep.",
      "Single seed (2026); exposed gold"),
    E("downstream-probe-both-block-v1.json", "probes", "adaptation", "probe", "current",
      "Gold probe: image+text last block vs image block alone, three seeds."),
    E("downstream-probe-cohort-compare-v1.json", "probes", "adaptation,text", "probe", "superseded",
      "Gold probe: original multilingual vs translated reports (image block).",
      "All results reproduced in downstream-probe-extended-v3"),
    E("downstream-probe-crop150-v1.json", "probes", "overfitting", "probe", "not-adopted",
      "Gold probe: 150 mm vs 130 mm crop, four blocks, three seeds."),
    E("downstream-probe-declip-v1.json", "probes", "encoders", "probe", "current",
      "Gold probe: DeCLIP vs the LoRA and full-FT bests.",
      "Exposed cohort; single seed"),
    E("downstream-probe-extended-v1.json", "probes", "adaptation", "probe", "superseded",
      "Gold probe: image-block language, depth and 40-epoch variants.",
      "Strict subset of extended-v3"),
    E("downstream-probe-extended-v2.json", "probes", "adaptation", "probe", "superseded",
      "Gold probe: adds four-block seeds, eight blocks, shuffled control.",
      "Strict subset of extended-v3"),
    E("downstream-probe-extended-v3.json", "probes", "adaptation", "probe", "current",
      "Canonical image-block variants gold probe, incl. shuffled-pairing control.",
      "Exposed gold (development statistic)"),
    E("downstream-probe-label-similarity-v1.json", "probes", "encoders", "probe", "superseded",
      "Gold probe: LM-label-agreement target (lambda 0.3) vs plain contrastive.",
      "Superseded by label-similarity-v2 probe"),
    E("downstream-probe-label-similarity-v2.json", "probes", "encoders", "probe", "null",
      "Gold probe: all four label-similarity arms on the unexposed cohort.",
      "Single seed"),
    E("downstream-probe-lora-holdout-v1.json", "probes", "adaptation", "probe", "current",
      "Gold probe: LoRA retrained without the gold studies (generalisation).",
      "Single seed; 433-candidate validation"),
    E("downstream-probe-plane-pooling-v1.json", "probes", "adaptation", "probe", "null",
      "Gold probe: uniform plane pooling vs learned attention, three seeds."),
    E("downstream-probe-unexposed-v1.json", "probes", "", "probe", "superseded",
      "Gold holdout probe: exposed vs unexposed four-block, three seeds.",
      "Superseded by unexposed-v2 (kept as the primary-result record)"),
    E("downstream-probe-unexposed-v2.json", "probes", "", "probe", "current",
      "Gold holdout probe plus the data-size control arm (authoritative)."),
    E("downstream-probe-v1.json", "probes", "", "probe", "current",
      "First probe: does contrastive learning beat unadapted DINOv2?"),
    E("downstream-probe-v1.md", "probes", "", "result", "current",
      "First probe write-up: per-label gains, plane-attention decomposition."),
    E("gold-holdout-comparisons-v1.json", "probes", "", "paired", "superseded",
      "Paired unexposed vs exposed / unadapted contrasts, per seed.",
      "v2 reproduces it exactly"),
    E("gold-holdout-comparisons-v2.json", "probes", "", "paired", "current",
      "Adds the data-size control: exposure vs data-loss decomposition."),
    E("gold-holdout-probe-protocol.md", "probes", "", "protocol", "current",
      "Pre-registration: does the gold-probe gain survive holding gold out?"),
    E("gold-holdout-v1.md", "probes", "", "result", "current",
      "How much was gold-probe AUROC inflated by training exposure?",
      "Quote the exposed level only beside the unexposed one"),
    # --- encoders ------------------------------------------------------------
    E("declip-alignment-results.md", "encoders", "", "result", "current",
      "Does DeCLIP (multi-view, NN queue, SimCLR) improve low-data alignment?",
      "Exposed cohort, off the holdout lineage; single seed",
      original="declip-alignment-v1.html"),
    E("declip-alignment-v1.html", "encoders", "", "report-html", "current",
      "Interactive legacy page: DeCLIP vs baselines.",
      "Exposed cohort; its 'record' language is not endorsed"),
    E("declip-features.npz", "encoders", "", "features", "current",
      "Cached probe features for DeCLIP and baseline arms."),
    E("label-similarity-paired-v1.json", "encoders", "probes", "paired", "superseded",
      "Paired gold-probe difference, lambda 0.3 minus lambda 0.",
      "Re-scored in paired-v2"),
    E("label-similarity-paired-v2.json", "encoders", "probes", "paired", "null",
      "Paired gold-probe differences for three label-similarity treatments."),
    E("label-similarity-v1.json", "encoders", "", "result", "superseded",
      "Curves, noise floor and probe summary for lambda 0 vs 0.3.",
      "Subsumed by label-similarity-v2.json"),
    E("label-similarity-v1.md", "encoders", "", "result", "null",
      "Does LM-label agreement as a soft contrastive target help?",
      "Single seed"),
    E("label-similarity-v2.json", "encoders", "", "result", "null",
      "Four-arm curves and noise floor for the label-similarity target."),
    E("label-similarity-v2.md", "encoders", "", "result", "null",
      "Can a smaller lambda or low-unknown labels rescue label similarity?",
      "Lambda 0.1 regularises the loss but not the features"),
    E("size-matched-holdout-features.npz", "encoders", "", "features", "current",
      "Cached probe features for the DINOv2-base / E5-small arm."),
    E("size-matched-holdout-probe.json", "encoders", "probes", "probe", "current",
      "Full gold-probe output for the DINOv2-base / E5-small arm."),
    E("size-matched-holdout-v1.html", "encoders", "", "report-html", "superseded",
      "Legacy two-arm backbone-swap page.",
      "Absorbed into the tower-size factorial"),
    E("size-matched-holdout-v1.json", "encoders", "", "result", "superseded",
      "Does DINOv2-base + E5-small (closer sizes) improve the gold readout?",
      "Absorbed into the tower-size factorial", original="size-matched-holdout-v1.html"),
    E("tower-size-factorial-curves.svg", "encoders", "", "figure", "current",
      "Train loss, val loss and val R@1 per epoch for four pairings."),
    E("tower-size-factorial-v1.html", "encoders", "", "report-html", "current",
      "Legacy page render of the tower-size factorial."),
    E("tower-size-factorial-v1.json", "encoders", "", "result", "current",
      "Machine-readable four-arm factorial with paired contrasts."),
    E("tower-size-factorial-v1.md", "encoders", "", "result", "current",
      "2x2 image/text tower size: effect on gold readout and overfitting?",
      "Single seed per arm", original="tower-size-factorial-v1.html"),
    E("tower-size-image-base-features.npz", "encoders", "", "features", "current",
      "Cached probe features, DINOv2-base / E5-base arm."),
    E("tower-size-image-base-probe.json", "encoders", "probes", "probe", "current",
      "Full gold-probe output, DINOv2-base / E5-base arm."),
    E("tower-size-text-small-features.npz", "encoders", "", "features", "current",
      "Cached probe features, DINOv2-small / E5-small arm."),
    E("tower-size-text-small-probe.json", "encoders", "probes", "probe", "current",
      "Full gold-probe output, DINOv2-small / E5-small arm."),
    # --- submission ----------------------------------------------------------
    E("kaggle-submission-pipeline.md", "submission", "", "protocol", "current",
      "How the Kaggle bundle is built and benchmarked, and what it scored.",
      "Public LB and OOF are different cohorts, not paired"),
    E("lm-head-tower-size-58-curves.svg", "submission", "", "figure", "current",
      "MLP head train/val masked BCE per epoch, three seeds x four pairings."),
    E("lm-head-tower-size-58-linear-curves.svg", "submission", "", "figure", "current",
      "Linear head train/val BCE per epoch, three seeds x four pairings."),
    E("lm-head-tower-size-58-v1.html", "submission", "", "report-html", "current",
      "Legacy page render of the 58-case LM-head comparison."),
    E("lm-head-tower-size-58-v1.json", "submission", "", "result", "current",
      "Machine-readable four-arm LM-head summary and paired size contrasts."),
    E("lm-head-tower-size-58-v1.md", "submission", "", "result", "current",
      "Do tower pairings differ under a matched LM-supervised head on 58 gold?",
      "58-case protocol; not paired with the 56-case numbers",
      original="lm-head-tower-size-58-v1.html"),
    E("lm-proxy-eval-v1.json", "submission", "", "raw-data", "current",
      "LM-verdict AUROC of the first three submitted heads, with public scores."),
    E("lm-proxy-eval-v2.json", "submission", "encoders", "raw-data", "current",
      "LM-test AUROC and paired intervals for the four tower-size LM heads."),
    E("lm-mlp-v1-results.json", "submission", "encoders", "raw-data", "historical",
      "Early machine-readable copy of the lm-mlp-v1 benchmark.",
      "Orphaned: unreferenced, differs from the checkpoint benchmark"),
    E("lm-mlp-v1.md", "submission", "encoders", "result", "current",
      "Does an LM-label-supervised MLP head beat linear and gold-only heads?",
      "56-case protocol; not a leaderboard improvement"),
    E("finetune-r4-multislice-design.md", "finetune", "submission", "protocol", "current",
      "R4 fine-tuning rung: K-window multi-slice input with per-label attention, decisions."),
    E("finetune-ladder-v1.json", "finetune", "submission", "result", "current",
      "Every fine-tuning step against its parent: LM-test, gold-58, paired intervals, curves.",
      "One training seed per step; gold-58 exploratory"),
    E("ensemble-selection-v1.json", "finetune", "submission", "result", "current",
      "Machine-readable ensemble-selection results: hand-picked combinations, direct comparisons, out-of-sample rules.",
      "All designs scored on the same 432 LM-test studies; gold-58 exploratory"),
    E("ensemble-selection-v1.md", "finetune", "submission", "result", "current",
      "Which fine-tuned runs to rank-average, and how to choose them, against u9 alone.",
      "Gold-58 is below u9 for every ensemble; none submitted"),
    E("leaderboard-history-v1.json", "submission", "finetune", "result", "current",
      "Every recorded Kaggle public-leaderboard submission with its local LM-test and gold-58 scores.",
      "Public split of the hidden test; one submission per bundle"),
    E("research/", "finetune", "encoders", "directory", "current",
      "Research notes, including the survey of MRI-pretrained backbones for fine-tuning."),
    E("next-model-candidates.md", "submission", "encoders", "protocol", "current",
      "Which tower swaps come next, and how they will be compared."),
    E("next-model-pairs-v1-contrastive-curves.svg", "encoders", "", "figure", "current",
      "Contrastive loss and both Recall@1 directions for the five tower swaps and the reference."),
    E("next-model-pairs-v1-linear-head-curves.svg", "encoders", "", "figure", "current",
      "Linear LM-head train/validation BCE per tower swap, three seeds."),
    E("next-model-pairs-v1-mlp-head-curves.svg", "encoders", "", "figure", "current",
      "MLP LM-head train/validation BCE per tower swap, three seeds."),
    E("next-model-pairs-v1.json", "encoders", "", "result", "current",
      "Machine-readable next-model-pairs results: retrieval, 58-gold and LM-test AUROC, paired intervals."),
    E("next-model-pairs-v1.md", "encoders", "submission", "result", "current",
      "Did any one-tower swap beat DINOv2-small + E5-base on retrieval, gold AUROC, or LM-test AUROC?",
      "One contrastive seed per tower; only RAD-DINO's gold interval excludes zero"),
    E("core-hypotheses-input-preprocessing.md", "pipeline", "submission", "result", "current",
      "Which image input and preprocessing hypotheses held up, and what remains open?",
      "Exploratory agent study; gold-58 exposed"),
    E("codec-declip-v1-answer.md", "encoders", "", "narrative", "current",
      "Hand-written answer section included in codec-declip-v1.md."),
    E("codec-declip-v1-contrastive-curves.svg", "encoders", "", "figure", "current",
      "Contrastive loss and both Recall@1 directions: reference, codec-pipeline replicate, DeCLIP arms."),
    E("codec-declip-v1-linear-head-curves.svg", "encoders", "", "figure", "current",
      "Linear LM-head train/validation BCE per arm, three seeds."),
    E("codec-declip-v1-mlp-head-curves.svg", "encoders", "", "figure", "current",
      "MLP LM-head train/validation BCE per arm, three seeds."),
    E("codec-declip-v1.json", "encoders", "", "result", "current",
      "Machine-readable codec-declip results: retrieval, 58-gold and LM-test AUROC, paired intervals, replicate noise, unadapted-tower anchors."),
    E("codec-declip-v1.md", "encoders", "submission", "result", "current",
      "Does the codec pipeline change training, and does DeCLIP beat plain CLIP?",
      "One contrastive seed per arm; replicate row measures run-to-run noise"),
    E("selected-kaggle-model.md", "submission", "", "result", "current",
      "Which exact model was submitted to Kaggle, with what settings."),
    # --- library: reviews & research notes -----------------------------------
    E("english-medical-text-encoder-research.md", "library", "", "research", "current",
      "Which English / medical sentence encoders to screen, and why."),
    E("english-medical-text-error-review-packet.md", "library", "", "review", "historical",
      "Reading packet: full paired report text for the 40 worst pairs.",
      "Evidence base for english-medical-text-error-review.md"),
    E("knee-clip-feedback.md", "library", "", "research", "current",
      "External critique: what the pipeline proves, and what to check."),
    E("knee-clip-feedback-response.html", "library", "", "report-html", "historical",
      "Legacy HTML rendering of the feedback response.",
      "Slightly stale vs the .md"),
    E("knee-clip-feedback-response.md", "library", "", "narrative", "current",
      "Point-by-point answers to the external feedback, with decision gates.",
      original="knee-clip-feedback-response.html"),
    E("knee-clip-feedback-source-notes.md", "library", "", "research", "current",
      "What MedCLIP, CLIP, E5 and OOF primary sources actually specify."),
    E("model-encoder-research.md", "library", "", "research", "superseded",
      "Which image and text encoders the project should start from.",
      "Recommendation superseded (reference is DINOv2-small + mE5-base)"),
    E("translation-api-primary-sources.md", "library", "", "research", "current",
      "What the Google Translate and Claude Code CLI docs guarantee."),
]

# Reading-room documents shown at the top of the Library page.
READING_ROOM = [
    "knee-clip-feedback.md",
    "knee-clip-feedback-response.md",
    "knee-clip-feedback-source-notes.md",
    "model-encoder-research.md",
    "english-medical-text-encoder-research.md",
    "english-medical-text-error-review-packet.md",
    "translation-api-primary-sources.md",
]

BY_PATH = {e.path: e for e in ENTRIES}

# Directories inside digest/ that are site output, not evidence.
OUTPUT_DIRS = {"site"}


def validate() -> None:
    problems = []
    seen = set()
    for e in ENTRIES:
        if e.path in seen:
            problems.append(f"duplicate catalog row: {e.path}")
        seen.add(e.path)
        if e.status not in STATUSES:
            problems.append(f"{e.path}: unknown status {e.status!r}")
        if e.kind not in KINDS:
            problems.append(f"{e.path}: unknown kind {e.kind!r}")
        for slug in e.pages:
            if slug not in BY_SLUG or slug == "index":
                problems.append(f"{e.path}: unknown page {slug!r}")
        if len(e.purpose.split()) > 15:
            problems.append(f"{e.path}: purpose longer than 15 words")
    if problems:
        raise SystemExit("catalog.py is inconsistent:\n  " + "\n  ".join(problems))


def check_complete(digest: Path) -> list[str]:
    """Fail the build when a digest/ file has no catalog row.

    Returns catalog rows whose file no longer exists (warnings, not errors).
    """
    validate()
    missing = []
    for item in sorted(digest.iterdir(), key=lambda p: p.name):
        if item.name in OUTPUT_DIRS or item.name.startswith("."):
            continue
        key = item.name + "/" if item.is_dir() else item.name
        if key not in BY_PATH:
            missing.append(key)
    if missing:
        raise SystemExit(
            "Build failed: these digest/ files have no catalog row. Add an entry to "
            "scripts/digest_site/catalog.py (page, kind, status, purpose) for each:\n  "
            + "\n  ".join(missing)
        )
    stale = [e.path for e in ENTRIES if e.path != "index.html" and not (digest / e.path.rstrip("/")).exists()]
    return stale


def for_page(slug: str) -> list[Entry]:
    return [e for e in ENTRIES if slug in e.pages]
