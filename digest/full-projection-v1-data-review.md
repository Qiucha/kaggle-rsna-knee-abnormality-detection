# Baseline data and report review

Continuation audit of the completed projection baseline, 2026-09-05.

**Decision: retain the baseline, but keep the pre-tuning review gate open.**
Cross-split duplicate report text is confirmed. Patient/image duplication is not
established by this audit. Do not silently deduplicate the frozen v1 cohort or
reinterpret the existing retrieval scores as patient-independent performance.

## Reproducible scope

Run from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-cuda/bin/python \
  scripts/audit_baseline_data.py --tokenizer-audit
```

[Machine-readable counts](full-projection-v1-data-audit.json) record the manifest,
rankings, script and label-file hashes. The script verifies the matrix-pinned
manifest checksum and each source/training report hash, and requires complete,
validation-only rankings in both directions. The ignored local
`checkpoints/contrastive/full-multilingual-v1/00-projection-full-mean-512/report-review-packet.json`
contains the 20 worst direction/query pairs with full query and top-candidate
reports, IDs, laterality and token counts. No inference or training runs here.
Test rows participate only in identity/hash overlap checks; no test performance,
report examples or per-class label support are reviewed/exported.

## Identity and duplicates

| Check | Duplicate groups | Studies in groups | Cross-split groups | Studies in cross-split groups |
| --- | ---: | ---: | ---: | ---: |
| Exact report text | 49 | 183 | 22 | 118 |
| NFKC + casefold + whitespace-normalized report | 54 | 204 | 23 | 125 |
| Study UID / observed patient group | 0 | 0 | 0 | 0 |
| Selected series path / metadata fingerprint | 0 | 0 | 0 | 0 |

Exact cross-split groups: nine train+validation, seven train+test, one
validation+test, five spanning all three. Normalization preserves punctuation,
numbers and negation; it does not establish semantic equivalence. Counts of
studies above include all members of each cross-split group.

The series fingerprints hash filenames, byte sizes and modification times
(`src/contrastive_learning/preprocessing.py:series_fingerprint`), **not pixel
content**. Zero overlapping fingerprints cannot rule out copied/re-encoded scans.
Observed patient grouping likewise cannot resolve anonymization semantics.
Exact report duplicates may be templates, repeated reports or repeated studies;
these possibilities require image/content and source-provenance investigation.

## Worst-pair report reading

All 20 packet entries were read. This is a provisional text comparison, not an
image interpretation or independent clinical adjudication. For report→image,
the competing image's paired report is only a proxy for what that image contains.
None of these 20 pairs has an identical normalized report. The table references
packet positions, with ranks from the saved baseline; it does not relabel positives.
Multilingual interpretations remain candidates for bilingual reviewer confirmation.

| Packet | Direction; positive rank | Report-level observation |
| --- | --- | --- |
| 1 | image→report; 303 | Query describes a tibial projection/pes-region issue and PCL/cartilage findings; French competitor describes Hoffa-region and soft-tissue edema with normal ligaments/menisci. Different reported findings. |
| 2 | image→report; 168 | Query findings mention a medial meniscal tear, but its conclusion says no other significant abnormality after discussing a ganglion. Competitor describes degeneration/OA and a Baker cyst. Internal query inconsistency merits source review. |
| 3 | image→report; 167 | Query describes bilateral effusion/synovitis with normal menisci; Greek competitor describes medial meniscal degeneration/tear and cysts. Bilateral report versus single-side study also needs review. |
| 4 | image→report; 154 | Query describes high-grade ACL disruption with preserved menisci; competitor says ACL normal and medial meniscal tear. Explicit disagreement. |
| 5 | image→report; 143 | Query describes medial meniscal tearing; Turkish competitor reports normal menisci and no increased joint fluid. |
| 6 | image→report; 122 | Shared absence of meniscal tear, but query emphasizes chondrosis and competitor a ganglion, marrow appearance and old tibial-tuberosity changes. |
| 7 | image→report; 120 | Shared language and some degenerative vocabulary; query describes ACL and medial meniscal rupture, competitor describes intact ACL and no medial meniscal rupture. |
| 8 | image→report; 103 | Query describes medial root tearing/extrusion; competitor explicitly reports no meniscal tear. |
| 9 | image→report; 100 | Both contain trauma/edema/ligament language, but query emphasizes patellar/MPFL injury while competitor emphasizes ACL and meniscal injury. |
| 10 | image→report; 90 | Partial overlap in medial meniscal and cartilage findings; reported ACL status differs. |
| 11 | report→image; 325 | Some shared degenerative/cyst vocabulary; competitor's report does not reproduce the query's leading tibial/pes finding and reports different meniscal/PCL findings. |
| 12 | report→image; 225 | Same internally inconsistent query as packet 2; competitor explicitly reports no meniscal tear. |
| 13 | report→image; 194 | Bilateral inflammatory report versus competitor's medial root tear/extrusion and chondrosis. |
| 14 | report→image; 160 | Shared ACL injury/effusion; competitor adds a displaced bucket-handle meniscal tear where query reports preserved menisci. |
| 15 | report→image; 154 | Shared medial cartilage degeneration/effusion; competitor describes meniscal maceration where query reports no tear. |
| 16 | report→image; 111 | Shared medial root tearing/extrusion and chondrosis; competitor additionally describes an insufficiency fracture and lateral tear. Similarity is partial. |
| 17 | report→image; 104 | Shared ACL injury; severe multicompartment OA/chronic ACL+PCL tearing differs from competitor's MCL injury, intact PCL and largely preserved tibiofemoral cartilage. |
| 18 | report→image; 99 | Closest thematic match: medial meniscal tearing with otherwise preserved structures. Tear morphology/extent differs; not established as interchangeable. |
| 19 | report→image; 96 | Acute ACL/lateral meniscal injury versus competitor's intact ACL/menisci and chronic cartilage/bursal findings. Shared long template does not imply shared findings. |
| 20 | report→image; 95 | Small medial meniscal tear versus competitor's normal menisci and chronic cartilage/bursal findings; same competitor as packet 19. |

The strongest supported follow-ups are report consistency, long-template
truncation, bilateral-report pairing and pixel-level review of the largest
outliers. These observations do not justify changing the objective, defining
multiple positives, or claiming that multilingual encoding caused the errors.

## Validation truncation

The pinned local E5 tokenizer (`query: ` prefix, special tokens included) counts
70/441 validation reports above 512 tokens. Seven of the 20 worst direction/query
entries exceed 512; repeated query studies appear in both directions, so these
are not seven independent studies. Length alone does not explain the worst
outlier: its query is 411 tokens. Two groups totaling five validation studies
have identical truncated model inputs, which can make unique paired-report
retrieval ambiguous. This is not a revised metric or proof of clinical equivalence.

## Label provenance and downstream support

Notebook cell 22 supplies a prompt to generate pathology likelihoods from reports.
Cells 23–24 call `data/cleaned.csv` binary gold and compare it to `data/lm_label/`.
That naming is not evidence of independent annotation: no annotator credentials,
image-versus-report source, adjudication record or annotation protocol was found
in those cells or the available CSV schema. All four LM files contain 58 distinct
cohort IDs; filenames identify model variants but do not preserve full generation
provenance. They remain report-derived weak supervision.

The manifest has labels for 48 training and eight validation studies. The other
3,478 training and 433 validation studies have empty label values, not negatives.
Validation MCL has eight negatives and zero positives, so discrimination metrics
requiring both classes are undefined. The other validation classes have only
2–6 positives each. No downstream fitting or test-label support analysis was done.
The existing downstream metric intentions remain provisional until the annotation
source and an adequately supported evaluation design are resolved.

## Remaining gate work

1. Hash decoded image content or verified preprocessed tensors, then review
   near-duplicate candidates on train/validation. Preserve the baseline and v1
   splits; any cohort/split change needs a new experiment version and baseline.
2. Investigate cross-split report duplicates as templates versus repeated studies;
   record a handling policy before launching matrix comparisons.
3. Confirm multilingual report interpretations and image/report correspondence for
   the worst outliers, including the contradictory and bilateral reports above.
4. Obtain label-annotation provenance and PatientID anonymization semantics from
   the dataset owner. Missing evidence remains an explicit limitation, not a pass.

M1 and M3 remain open. No matrix comparison, locked-test evaluation, commit or push
was performed in this continuation.
