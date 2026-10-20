# Digest index

Durable findings, protocols and audits for the knee MRI image–report contrastive project.
Start at the top; everything below is grouped by what it is for, not by when it was written.

## Navigable report site

Open [the research digest site](index.html) (it redirects to `site/index.html`). It follows the project's question, *which model to transfer?*, as a main track: [Phase 1 · contrastive pre-training](site/pages/phase1.html) → [Picking · which Phase-1 model to transfer](site/pages/picking.html) → [Phase 2 · label fine-tuning](site/pages/phase2.html) → [Kaggle submission](site/pages/submission.html). Three background pages hold the earlier work: [how the Phase-1 recipe was chosen](site/pages/recipe.html) (adaptation, tower sizes, probes), [overfitting & closed gates](site/pages/overfitting.html), and [pipeline, data & report text](site/pages/foundations.html). The [Evidence library](site/pages/library.html) lists every file in this folder with its page, kind and status. Every markdown document is also rendered inside the site under `site/docs/`. Old stage-page URLs (`pipeline`, `data`, `text`, `adaptation`, `probes`, `encoders`, `finetune`) redirect to the section's new page.

Rebuild from the repository root with `.venv/bin/python scripts/build_digest_site.py`. Run `.venv/bin/python scripts/build_digest_site.py export` first when run histories in `checkpoints/` have changed. The build fails if a file in `digest/` has no row in `scripts/digest_site/catalog.py`. The site works as local files; for HTTP serving, see [HOSTING.md](HOSTING.md).

**One-stage joint loss (2026-10-10).** The [Phase-2 results](site/pages/phase2.html#joint-loss) now include label BCE + 0.1 × image-to-report contrastive loss: LM-test 0.8809 vs label-only u9's 0.8834 (paired interval crosses zero), and exploratory gold-58 0.8494 vs 0.8775. See the [write-up](joint-loss-v1.md) and [snapshot](joint-loss-v1.json) for the matched u9 comparison, two-stage comparison, curves and per-label scores.

**Two numbers to carry everywhere.** The gold-probe level **0.7106** (pooled over three seeds; 0.7105
is seed 2026) is a *development statistic* read on studies the encoder trained on. **0.6938** is the
generalisation estimate on studies it never saw. Quote them together, never one instead of the other,
and never subtract the −0.0168 exposure gap from a *paired difference* — arms in a contrast share the
same exposure. See [gold-holdout-v1](gold-holdout-v1.md).

---

## Start here

| | |
| --- | --- |
| [project-state.html](project-state.html) | Generated run-status snapshot: retrieval curves, run telemetry, downstream probe card |
| [contrastive-learning.html](contrastive-learning.html) | The narrative page — what the project is, what moved the number, what is still open |
| [contrastive-learning-pipeline.md](contrastive-learning-pipeline.md) | How the pipeline works, end to end |
| [contrastive-learning-current-state.md](contrastive-learning-current-state.md) | Current-state audit of the implementation |

## Next model experiments

[next-model-candidates.md](next-model-candidates.md) lists the prioritized image and text tower swaps, model IDs, integration checks, and comparison protocol for the next experiment runner.

[finetune-r4-multislice-design.md](finetune-r4-multislice-design.md) records the decided design of the R4 fine-tuning rung (multi-slice input, per-label attention, fallback mask, memory and runtime budget).

## The main results, newest first

| | |
| --- | --- |
| [finetune-clip-init-u9-v1.md](finetune-clip-init-u9-v1.md) · [latest results](site/pages/phase2.html#latest-results) | Completed runs through 2026-10-17: A30 stays the best single LM-test run; the first ensemble, CLIP-init swap and deep three scored **0.894 → 0.899 → 0.902 public** |
| [ensemble-a30-v1.json](ensemble-a30-v1.json) · [digest charts](site/pages/phase2.html#a30-ensemble-sweep) | All 17 A30 and 12 CLIP-init combinations, 8 exportability checks, paired intervals, gold guard and Lateral Meniscus; deep three leads public |
| [omnirad-clip-frozen-probe-v1.md](omnirad-clip-frozen-probe-v1.md) · [digest page](site/pages/picking.html#frozen-probe) | Plain CLIP lifts the frozen OmniRad tower +0.043 LM-test (gold +0.027, CI spans zero); a frozen head still trails fine-tuning by 0.04 on gold; scanner identity also became easier to read |
| [combined-lora-k8-336-v1.md](combined-lora-k8-336-v1.md) · [digest page](site/pages/phase1.html#combined-lora-k8-336) | Combined CLIP + LoRA + K8 + 336, both arms complete: OmniRad-base 0.8812 LM-test / 0.7844 gold-58, DINOv2-small 0.8670 / 0.7652 (frozen MLP heads) |
| [run5-followups-v1.md](run5-followups-v1.md) · [digest page](site/pages/phase2.html#run5-followups) | Completed run-5 follow-ups, including label fine-tuning from the combined LoRA checkpoint, paired comparisons, and remaining queue |
| [factor-isolation-v1.md](factor-isolation-v1.md) · [digest page](site/pages/phase1.html#factor-isolation) | Completed contrastive arms, gold/LM-test heads, reference and matched-control deltas with paired intervals, training curves and per-label heatmaps; current evaluation status in the report |
| [lm-head-tower-size-58-v1.md](lm-head-tower-size-58-v1.md) · [HTML](lm-head-tower-size-58-v1.html) | Full five-row settings matrix: original 56-case gold-only submission plus four 58-case LM-head runs (**0.7659, 0.7541, 0.7504, 0.7605** AUROC); paired size intervals include zero |
| [crop150-probe-v1.md](crop150-probe-v1.md) | **The 150mm crop does not help the labels either: −0.0201 macro AUROC**, same sign 3/3. Costs retrieval AND label readout — the crop is not adopted |
| [budget-40ep-v1.md](budget-40ep-v1.md) | **The budget arm.** Both arms retrained at 40 epochs: drop-path 0.1 gained +0.0150 vs the baseline's +0.0062, deficit −0.0255 → **−0.0167**, CI now grazes zero. Pre-registered rule: **ambiguous** |
| [convergence-curves.html](convergence-curves.html) | **Convergence audit of both 2026-09-15 results.** Drop-path arms were **truncated mid-improvement** (rate 0.1 still falling at epoch 20, 3/3 seeds); crop150 **converged and overfit**, so its reading stands |
| [crop150-rebaseline-v1.md](crop150-rebaseline-v1.md) | **New preprocessing baseline.** 150mm crop retains 96% of tissue vs 86%; retrieval 0.1340 → 0.1190, seed spread 4× tighter (convergence-checked; **label readout also −0.0201, not adopted**) |
| [drop-path-sweep-v1.md](drop-path-sweep-v1.md) | **Stochastic depth, six rates over 20×: no knee.** The memorisation/retrieval trade is parallel — but **amended 2026-09-15**: closed *at a 20-epoch budget*, since the high rates had not converged — and the budget arm then recovered a third of the rate-0.1 deficit |
| [drop-path-5e-2-arm.md](drop-path-5e-2-arm.md) | Stochastic depth at 5e-2: memorisation gap cut **41%**, retrieval **−0.0153**. Less memorisation did not buy generalisation |
| [gold-holdout-v1.md](gold-holdout-v1.md) | **The generalisation estimate, 0.6938.** Exposure inflated the headline by −0.0168; +0.0500 of the gain is real |
| [matched-n-loss-audit.md](matched-n-loss-audit.md) | The image tower overfits: train R@1 **3.0×** validation at matched N |
| [image-block-variants-v1.md](image-block-variants-v1.md) | Depth is the lever: 4 blocks beats 1 at three seeds. Language and schedule are null |
| [plane-pooling-v1.md](plane-pooling-v1.md) | Uniform vs learned plane pooling — null by the pre-registered rule, and underpowered |
| [downstream-probe-v1.md](downstream-probe-v1.md) | The first downstream probe |
| [downstream-both-block-v1.md](downstream-both-block-v1.md) · [downstream-original-vs-translated-v1.md](downstream-original-vs-translated-v1.md) | Both-tower adaptation; original vs translated reports |

## Data and preprocessing audits

| | |
| --- | --- |
| [acquisition-metadata-trust.md](acquisition-metadata-trust.md) | **What the series CSV cannot tell you.** The two flags are one bit; field of view, depth sampling, slot fill, laterality, protocol leak |
| [gold-case-inventory.md](gold-case-inventory.md) | Provenance and exposure of the 58 gold studies |
| [integrity-coverage-audit.md](integrity-coverage-audit.md) | Duplicates, joins, split independence |
| [duplicate-confusion-audit.md](duplicate-confusion-audit.md) | Do duplicate reports explain retrieval misses? |
| [image-shape-audit.md](image-shape-audit.md) | Image tensor shapes |

## Gates — checks that closed a direction before it cost GPU time

| | |
| --- | --- |
| [false-negative-rate-audit.md](false-negative-rate-audit.md) | Rules out false-negative cancellation and soft targets at batch 32 |
| [val-loss-vs-retrieval-audit.md](val-loss-vs-retrieval-audit.md) | Validation loss and retrieval disagree — **carries a dated retraction, read it** |
| `hard-negative-gate-data.json` | Lever A fails at batch 32: the top-1 negative already holds 0.73–0.75 of the denominator |
| `protocol-leak-check.json` | Series count predicts none of the 12 gold labels |

## Protocols — pre-registrations, written before their runs

[gold-holdout-probe-protocol.md](gold-holdout-probe-protocol.md) ·
[plane-pooling-protocol.md](plane-pooling-protocol.md) ·
[claude-translated-dfd07db-image-block-protocol.md](claude-translated-dfd07db-image-block-protocol.md) ·
[claude-translated-dfd07db-text-block-protocol.md](claude-translated-dfd07db-text-block-protocol.md) ·
[english-medical-text-protocol.md](english-medical-text-protocol.md)

## Research notes and reviews

[model-encoder-research.md](model-encoder-research.md) ·
[english-medical-text-encoder-research.md](english-medical-text-encoder-research.md) ·
[knee-clip-feedback.md](knee-clip-feedback.md) ·
[knee-clip-feedback-response.md](knee-clip-feedback-response.md) ·
[knee-clip-feedback-source-notes.md](knee-clip-feedback-source-notes.md) ·
[translation-api-primary-sources.md](translation-api-primary-sources.md)

## Historical — kept as the record of how results were obtained

[full-projection-v1-results.md](full-projection-v1-results.md) ·
[full-projection-v1-handoff.md](full-projection-v1-handoff.md) ·
[full-projection-v1-data-review.md](full-projection-v1-data-review.md) ·
[claude-translated-dfd07db-results.md](claude-translated-dfd07db-results.md) ·
[claude-translated-dfd07db-comparison-results.md](claude-translated-dfd07db-comparison-results.md) ·
[claude-translated-dfd07db-continuation.md](claude-translated-dfd07db-continuation.md) ·
[claude-translated-dfd07db-image-block-results.md](claude-translated-dfd07db-image-block-results.md) ·
[claude-translated-dfd07db-image-block-error-review.md](claude-translated-dfd07db-image-block-error-review.md) ·
[english-medical-text-results.md](english-medical-text-results.md) ·
[english-medical-text-error-review.md](english-medical-text-error-review.md) ·
[translation-update-dfd07db.md](translation-update-dfd07db.md) ·
[cleanup-reproduction-20260908.md](cleanup-reproduction-20260908.md) ·
[experiment-session-log.md](experiment-session-log.md) ·
[config-translation-review.md](config-translation-review.md) ·
[claude-translated-baseline.md](claude-translated-baseline.md) ·
[contrastive-learning-implementation-status.md](contrastive-learning-implementation-status.md) ·
[english-medical-text-error-review-packet.md](english-medical-text-error-review-packet.md)

---

## Conventions that keep these documents honest

1. **Never quote `train_loss` beside `validation_loss` as a train/validation pair.** One is in-batch
   over 32 with augmentation on; the other is the full 441-matrix unaugmented. For a regularisation
   arm they are not even comparable *between arms* — drop-path is active in the logged train pass and
   absent in the baseline's. Use the matched-N route on eval-mode embeddings.
2. **Never put retrieval `pp` figures in a table with gold-probe AUROC deltas.** Different
   populations, different metrics.
3. **Candidate counts must match.** Validation R@1 is over 441 on the exposed cohort and 433 on the
   gold-holdout cohort. They never share a table.
4. **Compare losses only at a matched inverse temperature.** τ scales all logits, so it moves loss
   without moving retrieval. R@1 is τ-invariant and needs no correction.
5. **Selection is retrieval R@1, always.** Nothing in this project has ever been selected on label
   AUROC.
6. **The test split has never been loaded.**
