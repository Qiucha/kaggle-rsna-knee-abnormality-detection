# English / medical English text replacement — frozen first screen

Frozen 2026-09-07 before any new candidate retrieval scores. Version:
`english-medical-text-v1`. Research and preparation are complete once the linked
preflight artifact passes; this document does not claim training has run.

## Question and shortlist

Does replacing multilingual E5's frozen report features improve exact-pair
MRI/report retrieval on the existing Claude-translated cohort? Compare exactly:

| Candidate | Revision | Feature extraction | Prefix |
| --- | --- | --- | --- |
| `intfloat/e5-base-v2` | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | Attention-mask mean, 768 features | `query: ` |
| `NeuML/pubmedbert-base-embeddings` | `b79526d6ef3645e0df4530322e266f24c829f5ef` | Attention-mask mean, 768 features | Empty |

See [primary-source research](english-medical-text-encoder-research.md) for domain,
training objectives, licenses, dependencies and capacity. PubMed scientific
sentence training is not knee-radiology-specific validation. Existing RadBERT and
BiomedBERT pilot configs were inspected: five epochs, 256-token cap, unpinned
revisions, and historical pooling defaults. They are not matched performance
references and did not determine this shortlist.

Both candidates use standard BERT `encoder.layer` paths supported by the existing
freeze implementation. Neither requires custom code or a pretrained Dense
embedding head. The E5 sentence package normally L2-normalizes mean features for
text cosine similarity. Here raw mean features enter the existing learned
LayerNorm → Linear → GELU → Linear → L2 task head, exactly as in the multilingual
E5 baseline. This explicitly chooses pre-normalization backbone features; it does
not reproduce complete SentenceTransformers outputs or claim LayerNorm equals L2
normalization. NeuML's sentence package is Transformer → mean Pooling.

Both hidden widths are 768. Each text projection has 264,192 parameters; total
trainable parameters stay 827,906 (image fusion/projection, text projection and
logit scale). Tokenizers, vocabulary/backbone size and pretrained training differ.
This is a comparison of declared encoder packages, not literally weights alone.

## Fixed controls and references

[Machine-readable protocol and resolved configs](../configs/contrastive/english-medical-text-v1.json)
copy every setting from the saved quarantined seed-2026 baseline. The only screen
config differences are text ID/revision/prefix and output directory. Pooling is
mean in all arms. Current training source matches both prior execution snapshots
byte for byte. Source and baseline-config SHA-256 values are frozen in that file.
The independent six-run verifier was rerun successfully during this continuation;
it verifies stored reload evidence, not a newly executed GPU reload.

Use manifest `data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl`, SHA-256
`8f6f021954e268b995dbe63daf42a2cc5f0dd92efdd27ad511b0ddbe5832f99b`.
Preserve 3,524 eligible train and 441 validation studies, full reports and
`translated_english` metadata. Retain source-English alongside translations.
Both quarantined rows remain ineligible. Test stays locked.

DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` and
both backbones stay fully frozen. Fresh pretrained initialization for every run;
no old projection checkpoint grafting. Keep 20 epochs, batch 32, seed 2026,
512-token cap, 256 shared dimensions, AMP, augmentation, 224-pixel images,
130-mm crop, three slices per plane, optimizer, weight decay, clipping, warmup
and cosine schedule unchanged. Full resolved values are in the JSON; do not
reconstruct them from historical CLI defaults. Preserve the existing AMP behavior
and disclose prior epoch-2 infinite gradient norms and missing skipped-step counts.

Reuse frozen multilingual E5 baseline references at seeds 2026/2027/2028 only
while source/config/data compatibility continues to hold. Their respective
selected mean R@1 values are 10.6576%, 8.9569%, 8.8435%; all checkpoint hashes and
paths are included in the JSON. Do not compare the screen to image-adapted E5.
If compatibility fails, stop and version a revised matched protocol before scores;
additional baseline training is not silently added to this budget.

## Preflight and finite execution

1. Verify local metadata against pinned primary files and run
   `.venv/bin/python scripts/audit_text_encoder_preflight.py`. It checks the
   protocol/source/cohort/config and audits eligible train/validation reports
   only: tokens, truncation, retained characters, late summary headings, and
   empty/invalid token outputs. See [audit](english-medical-text-preflight.json).
   Heading visibility is not a clinical completeness measure.
2. Preserve the outstanding image-adaptation candidate error/subgroup review
   from the prior handoff before launching new training. It remains an unfinished
   evidence item, not evidence against either text candidate.
3. Download only the two pinned candidate snapshots, retaining tokenizer/config
   files and weight hashes. No paid service. Model metadata/tokenizers are already
   local; weights have not been downloaded. Freeze a new execution source copy,
   dependency inventory and an exclusive execution claim; do not reuse an old
   completed queue launcher or overwrite completed artifacts.
4. One GPU smoke per candidate: seed 2026, batch 32, one epoch, at most 64 studies,
   zero workers. Verify both backbones frozen/eval, finite nonzero normalized
   256-dimensional outputs, expected trainable counts, actual peak GPU memory,
   and exact compact-checkpoint validation reload. Record runtime/memory and
   every failure. Smoke scores are infrastructure checks, not selection evidence.
   An OOM, pooling mismatch or reload discrepancy stops that candidate; do not
   silently lower its batch/length, change precision or replace it with a third
   candidate. A fix requires a recorded deviation and new attempt directory.
5. Run E5 then NeuML at seed 2026, each with all 3,524 training studies and 441
   validation candidates. Select mean bidirectional R@1 descending, validation
   loss ascending, earliest epoch. Verify all 20 history rows, selected checkpoint
   SHA-256 and exact validation reload before interpreting the result.
6. Repeat each successful candidate at seeds 2027 and 2028 only if its selected
   seed-2026 R@1 is strictly greater than the matched frozen E5 baseline. Maximum:
   two smoke runs and six full candidate runs. No automatic image adaptation,
   text unfreezing, report-view variants, third model, or further seed search.

## Completed tokenizer preflight

The audit passed for all 3,524 eligible training reports and 441 validation
reports, with zero empty reports or invalid token outputs in all three arms.

| Encoder | Train reports over 512 | Validation reports over 512 | Validation last summary heading not fully retained |
| --- | ---: | ---: | ---: |
| Multilingual E5 baseline | 464 | 67 | 43 |
| English E5-base-v2 | 343 | 45 | 22 |
| PubMedBERT Embeddings | 141 | 19 | 4 |

There are 264 validation reports with an impression/conclusion heading under the
lexical heuristic. Lower truncation is an observed tokenizer difference, not
retrieval evidence. The shortlist and comparison controls were fixed before this
audit; no candidate retrieval scores were viewed. The first audit attempt was
interrupted for inefficient repeated tokenizer vocabulary-size calls; caching
that invariant completed the same audit without changing its input contract.

## Reporting and next decision

Report all attempts, failures, selected epochs, both retrieval directions,
R@1/5/10 and rank metrics, runtime and measured peak memory. For each available
seed, pair both directions by the same 441 study UIDs; average directional top-1
credit per study, then candidate minus baseline. Report 10,000 study-paired
bootstrap draws (fixed seed 2026), mean difference, 95% descriptive interval,
and across-seed mean/sample SD/range when three seeds exist. Keep exact-pair
single-positive loss and uniform expected tie credit unchanged.

Per-query intervals do not correct candidate/checkpoint selection, multiple
comparisons or repeated patients. Review errors and train/validation language,
length and summary-visibility groups with the full 441-candidate pool. Do not
infer diagnostic accuracy or independent-patient performance. Unknown patient
identity, duplicate/template reports, translation uncertainty and weak labels
persist. No locked-test inspection or downstream fitting.

After text-swap results and error review, decide whether a promising candidate
merits a separately declared comparison under the same image-final-block setting.
There is no automatic final-model selection from this screen.
