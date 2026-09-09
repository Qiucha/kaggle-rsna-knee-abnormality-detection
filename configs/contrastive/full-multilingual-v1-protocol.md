# Frozen experiment protocol: full-multilingual-v1

Frozen on 2026-09-05. The source revision is the Git commit introducing this file;
run manifests record the actual commit and source-tree hash used for each run.
The matrix, dependency lock, artifact inventory, and source defaults at that
revision jointly define the experiment. Changes require a new version and output root.

## Cohort and selection

The checksum-pinned cohort has 4,407 eligible studies: 3,526 training, 441 validation,
and 440 locked test. Use raw multilingual reports with the exact pinned E5 and
DINOv2 revisions in the matrix. No pathology labels enter contrastive training.
Select checkpoints by maximum bidirectional mean validation Recall@1, then minimum
validation loss, then earliest epoch. Across experiments, identical recall/loss
uses declared matrix order. Validation loss is symmetric contrastive loss over the complete validation
embedding matrix; keep validation batch size/order fixed across comparisons.
Training, resume, and finalization share this rule and verify saved checkpoint
metrics. Tied retrieval similarities receive expected credit as implemented and tested.

The primary baseline freezes both backbones, trains fusion/projection/temperature,
uses full reports, mean text pooling, a `query: ` prefix, 512 tokens, seed 2026,
batch size 32, and 20 epochs. Remaining defaults are fixed by the versioned
`TrainConfig` and CLI. Do not tune against the locked retrieval test split.

Review both retrieval directions, loss curves, temperature, embedding standard
deviation, paired/unpaired similarities, and per-query failures. For 441 validation
candidates, random Recall@1/5/10 is 1/441, 5/441, 10/441 (approximately 0.227%,
1.134%, 2.268%) and expected rank is 221. Report failures and negative results.
A continue/revise decision requires a completed baseline and diagnostics. The baseline
completed on 2026-09-05; retain it as the comparison anchor. The pre-tuning gate
remains open: see `digest/full-projection-v1-data-review.md` for confirmed cross-split
report duplicates, provisional report-error review and unresolved image/provenance checks.

## Leakage and label limitations

The reader enforces unique study IDs and disjoint observed patient groups across
splits. Patient IDs are hashed with a fixed domain prefix; missing IDs fall back
to study identity. The prior header audit found every observed PatientID unique.
Anonymization semantics are unavailable, so this cannot prove patient independence.
Duplicate or clinically equivalent reports and images can still create false
negatives or leakage; audit these before representation tuning.

The 12 CSV label columns and their value counts are recorded in the artifact
inventory. The notebook calls 58 rows ground truth, but independent human annotation
provenance has not been established. The remaining 4,349 rows lack those labels.
Files under `data/lm_label/` are model-derived labels and must be analyzed as weak
supervision separately; do not promote them to independent clinical reference labels.

Before downstream fitting, resolve provenance and label support under the fixed
splits. Compare frozen contrastive image features with unadapted DINOv2 using the
same classifier budget. Predeclare per-label and macro/micro AUROC/AUPRC,
validation-selected threshold metrics, calibration, and bootstrap uncertainty;
mark undefined metrics when a split lacks positives or negatives. No downstream
performance claim or downstream split redesign is authorized by these pilot labels.
