# Integrity coverage: duplicates, joins, and split independence

Read-only audit, 2026-09-11 (Asia/Taipei). Answers the open half of
[the feedback's §7](knee-clip-feedback.md) — exhaustive duplicate and join coverage,
and patient independence across the whole cohort — which
[the response §8](knee-clip-feedback-response.md) left as the largest unfinished P0 item.

Cohort: `data/processed/contrastive-cohort-claude-dfd07db-q2-v1.jsonl`, 4,407 studies
(3,526 train / 441 validation / 440 test). Evidence:
[`integrity-coverage-audit.json`](integrity-coverage-audit.json),
[`acquisition-duplicate-screen.json`](acquisition-duplicate-screen.json),
[`duplicate-group-pixel-verification.json`](duplicate-group-pixel-verification.json).
Scripts: `scripts/audit_integrity_coverage.py`,
`scripts/audit_acquisition_duplicates.py`, `scripts/verify_duplicate_groups.py`.

No pixel of the locked test was decoded and no test label or report was read. Test
studies appear here only as split names and counts.

## What changed

Two things, and the second matters more than the first.

**Duplicate coverage now spans the whole cohort instead of one verified pair** — for
duplicates of the kind the screen can see, which is a real restriction stated under
"What is still not covered". Within that reach the answer survives: **exactly one
pixel-identical duplicate group exists in the entire cohort**, the one already
quarantined. Six further groups the screen flagged were decoded and hold *different*
images — including the train/test candidate, which was gated on a user decision, taken on
2026-09-11, and cleared.

**Patient independence is not merely unresolved — it is unverifiable from this data,
and there is a positive control proving it.** The two studies of the known duplicate
pair have identical decoded pixels and *different* `PatientID` and
`FrameOfReferenceUID` values. De-identification re-randomised the identity tags per
study. So the cohort-wide result below — 4,407 distinct `PatientID` values over 4,407
studies — is an artefact of anonymisation, not evidence that every study is a different
patient. It cannot be read as independence.

## Patient independence

| Check | Result |
| --- | --- |
| `patient_group_id` distinct values | 4,407 over 4,407 rows |
| Group size distribution | every group has exactly one study |
| Groups spanning splits | 0 — vacuously, since no group has two members |
| `PatientID` present | 4,407 of 4,407 studies, **4,407 distinct** |
| `PatientID` groups spanning splits | 0 |
| `FrameOfReferenceUID` distinct | 4,407 |
| `PatientName`, `PatientBirthDate`, `StudyDate`, `StudyTime`, `AccessionNumber` | absent from every study |
| `InstitutionName`, `DeviceSerialNumber` | absent from every study |

The manifest's grouping key is a per-study pseudonym, so grouped splitting is not
merely unused — there is nothing to group by. The split is study-level random. If a
patient contributed two studies, nothing in this repository could show it, and the
split would separate them without anyone noticing.

The known duplicate pair is the proof. Two studies whose selected series decode to
byte-identical pixels carry two different `PatientID`s and three pairs of different
`FrameOfReferenceUID`s. Whatever the true patient relationship, the identifiers do not
express it.

**Consequence for reported numbers.** Every retrieval figure on this cohort is a
study-level, not patient-level, estimate, and cannot be described as patient-disjoint.
This is a limitation to state, not a defect to repair here: recovering patient identity
needs upstream data the competition does not supply.

## Duplicate coverage

Prior coverage was 3,967 cached train+validation tensors under a 16×16 pooled
near-duplicate screen, plus exact-pixel verification of one known pair. Test was
uncovered, and raw volumes were uncovered. `/mnt/d/train_series` is **532 GB** over
4,407 studies on a 9p-mounted Windows drive, so hashing raw volumes is out of reach;
test pixels were gated at the time. The screen used here reads *headers* instead; one
test study was decoded later, under a user decision, to settle the single candidate that
headers could not.

It runs in two stages, and it is validated by a positive control before its null
results are believed.

1. **Stage 1** hashes series-constant acquisition tags (matrix, spacing, TR/TE, coil,
   scanner, protocol name) plus slice count, for each of the three selected planes.
   One header read per series; 4,407 studies in 4.9 minutes.
2. **Stage 2** compares, within each stage-1 collision group, the sorted multiset of
   `SliceLocation` values per plane.

Stage 2 exists because of a failure worth recording. A first version hashed the
*middle file's* geometry and **missed the known duplicate**. Filenames are randomised
SOP UIDs, so name-sorting picks a physically different slice from each copy: the two
copies differ in `InstanceNumber`, `SliceLocation`, `ImagePositionPatient` and window
values for that reason alone. Sorting slice positions removes the dependence on file
naming. The known pair is retained as a positive control — with the fix it is recovered
at both stages, which is what makes the null results below meaningful.

| Stage | Result |
| --- | --- |
| Stage 1 candidate groups | 11 groups, 31 studies, largest 7 |
| Stage 2 confirmed acquisition groups | 7 groups, 19 studies |
| Groups crossing splits | 2 (one train+validation, one train+test) |
| Positive control recovered | yes, at both stages |

Decoding settles them. Six groups contain no test study and were decoded in full —
every slice of each selected series, hashed in the pipeline's own geometric order:

| Group | Splits | Pixels identical | Verdict |
| --- | --- | --- | --- |
| 7 studies | train + validation | no, on all three planes | image-duplicate **negative**, independence **unresolved** |
| 2 studies ×4 | train | no | image-duplicate negative, independence unresolved |
| 2 studies | train | **yes, all three planes** | the known quarantined pair |
| 2 studies | **train + test** | no, on all three planes | **cleared** — decoded 2026-09-11 under a user decision |

So the response's "one duplicate pair" conclusion holds for train and validation, and
now rests on cohort-wide rather than single-case coverage. Both members of that pair are
already `training_eligible: false` in the quarantined cohort.

**The six cleared groups are cleared of the wrong charge.** They are not duplicate
images. But §7's other question is patient independence, and a second scan of the same
patient on the same scanner under the same protocol produces precisely this signature:
identical series-constant tags, identical slice count, identical sorted slice positions,
different pixels, different reports. Because patient identity is unrecoverable by every
other route, **these 19 studies are the only patient-level leakage candidates this
cohort can produce.** Two of the groups cross a split: the 7-study group spans train and
validation, and the now-cleared pair spans train and **test**. Cleared of duplication is
not cleared of independence — if those two studies are one patient, the test split is
carrying a training patient, and nothing in this data can rule that in or out.

One cheap discriminator was run and did not settle it. Two timepoints of one patient
usually share unusual specifics, so report similarity should be elevated. Within the
7-study group mean pairwise similarity is 0.125 and the maximum 0.232, against 0.031
mean for 3,000 random train/validation pairs — above the random 95th percentile (0.099)
but below its 99th (0.308). The four train-only pairs range from 0.042 to 0.307. That is
suggestive of shared template language at most, and it neither confirms nor excludes
repeated patients. Recorded so a future reader does not re-run it expecting an answer.

**The train/test candidate was gated, then decoded, and it is cleared.** One candidate
group paired a train study with a test study on identical acquisition tags and identical
slice geometry, which only decoded pixels could settle. The response §8 reserves that
decision — a whole-dataset integrity check involving test needs a predefined separation
from outcome and model-selection access — so it was deferred and the user took it on
2026-09-11.

The decode kept a deliberately narrow boundary: the test study's **pixels only**. Its
report text was never read, its labels were never read, no model was loaded and no metric
computed. The two reports were compared by manifest sha256, which establishes agreement
without either report being exposed. So the only information crossing the split boundary
is one integrity verdict, and nothing that could inform model selection.

**Result: not a duplicate.** Decoded pixels differ on all three planes, and the reports
differ too. Same scanner, same protocol, same slice geometry, different patient encounter.
The test split is unaffected, and **no candidate group remains gated**. Evidence:
[`gated-duplicate-resolution.json`](gated-duplicate-resolution.json). Script:
`scripts/decode_gated_duplicate_candidate.py`.

## Join coverage

| Check | Result |
| --- | --- |
| `train_series.csv` rows | 24,371 |
| Distinct `SeriesInstanceUID` | 24,371 |
| Series appearing under more than one study | **0** |
| Manifest slot references (3 planes × 4,407) | 13,221 |
| Distinct series directories referenced | 13,221 |
| Directories shared by more than one study | **0** |

The image–report join is one-to-one at both the source table and the manifest. There is
no many-to-one join, so the known duplicate cannot be explained by two studies pointing
at one series: they point at different directories holding equal pixels.

## Duplicate report text

Exact text collisions, whole cohort:

| | Groups | Studies | Largest group | Groups crossing splits | Studies in crossing groups |
| --- | --- | --- | --- | --- | --- |
| Translated report | 44 | 135 | 10 | 20 | **78** |
| Source report | 49 | 183 | 37 | 22 | 118 |

The translated figure reproduces the previously reported "20 exact groups spanning
splits, 78 studies" exactly — that figure counted only the split-crossing subset; the
full count is 44 groups over 135 studies. The source-report count is larger, as
expected, because translation collapses some wordings and separates others.

**These are templates, not duplicate patients.** The three largest source groups are
a one-line entirely-normal knee template (37 studies), a Hoffa's fat pad impingement
template (14) and a one-word unremarkable template (12). And the cross-check is clean: of the 135 studies in translated
text-collision groups and the 183 in source-report groups, **zero** appear in any
acquisition-duplicate candidate group. Repeated text and repeated images are disjoint
phenomena in this cohort.

They still matter for retrieval. A query whose caption is shared by nine other studies
has no unique correct target, so R@1 is bounded below 100% by construction on those
studies. That is an interpretation caveat for the retrieval numbers, separate from
integrity.

## Series fingerprints: zero collisions, and why that is uninformative

`series_fingerprints` collide on no plane and on no study, but the fingerprint hashes
`PREPROCESSING_VERSION` plus each file's **name, size and mtime**
(`src/contrastive_learning/preprocessing.py:196`) — stat metadata, not content. It
cannot see a re-exported duplicate, and its zero-collision result is consistent with
the known pixel-identical pair, which it does not detect. Reported for completeness.
The existing notes in [the projection data review](full-projection-v1-data-review.md)
and the archived handoff already state this limitation correctly; no document was found
claiming duplicate coverage on the strength of fingerprints.

## Upstream root cause of the known pair: closed as unrecoverable

The response accepts "explicitly document why it cannot be recovered" as satisfying the
closure criterion, and that is the outcome. What can be said:

- both copies are SIEMENS Aera acquisitions with identical protocol names
  (`t2_pd_fs_TRA`, `pd_tse_fs_cor_320`, `pd_tse_fs_sag_320`), identical matrices and
  identical slice counts (30 / 27 / 30);
- both are tagged laterality `R` in the manifest, while one report describes a left
  knee;
- every identifier that could link them to a common origin — study, series, SOP, frame
  of reference, patient — differs, and no acquisition date, institution or device
  serial survives de-identification.

There is therefore no evidence in this repository that can say which report belongs to
the acquisition, or whether the duplication happened at acquisition, export, or
identifier assignment. Quarantine contains the contradiction; it does not resolve it,
and neither does this audit.

## What is still not covered

- **Duplicates that differ on any stage-1 tag.** Stage 1 is a *conjunctive* hash over 25
  series-constant tags plus slice count, so a single differing value — a re-derived
  `SeriesDescription`, a changed `SoftwareVersions` string, a re-exported
  `ReceiveCoilName` — makes a pair invisible. The positive control has n=1 and happens
  to agree on all 25, which demonstrates sensitivity to clean re-export with full tag
  preservation and to nothing else. The screen's reach is "duplicates that agree on
  every stage-1 tag", not "duplicates".
- **Raw volumes.** Only the three selected series per study were screened, and only by
  header. A duplicate that differs in the selected series but repeats elsewhere in the
  532 GB of raw data would not appear here.
- **The locked test's pixels**, except for the one study decoded to settle the train/test
  candidate. The other 439 test studies are covered by acquisition headers only.
- **Near-duplicates.** The screen finds equal acquisitions, not similar ones. The
  existing 16×16 pooled screen remains the only near-duplicate evidence, and it covers
  train and validation only.
- **True patient identity**, which no amount of local analysis can recover.
