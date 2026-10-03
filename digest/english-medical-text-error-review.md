# English / medical text encoder error and subgroup review

Qualitative review of the two completed frozen-text-encoder screens against the
multilingual-E5 baseline, on the quarantined Claude-translated full-report cohort
(`dfd07db`), seed 2026, frozen DINOv2 image tower. It closes the review item left
open by [the results](english-medical-text-results.md) and
[the error-review packet](english-medical-text-error-review-packet.md).

All forty pairs in the packet — twenty per candidate — were read in full. Ranks,
provenance, token counts and group labels quoted here are taken from the packet's
own pair headings; the composition counts below were recomputed from them.

No test data, no downstream fitting, no clinical adjudication of image pixels, and
no change to the cohort, the single-positive objective, the selection rule or the
locked test. In `text_to_image` rows the comparison text is the retrieved *image's*
paired report, not an interpretation of that image.

## What the two screens measured

| Candidate | Selected epoch | Mean R@1 | Gain vs baseline pp | Descriptive paired 95% CI pp |
| --- | ---: | ---: | ---: | --- |
| e5-base-v2 | 12 | 8.8435% | -1.8141 | [-4.4218, 0.6803] |
| pubmedbert-base-embeddings | 19 | 9.4104% | -1.2472 | [-3.9683, 1.3605] |

Both intervals include zero, both candidates are one seed, and neither triggered a
conditional repeat. Nothing below changes that: this review explains *where* the two
candidates spend their errors, not whether either is better than the baseline.

## The twenty-worst sets are not a like-for-like comparison

The two sets are disjoint rankings over different queries. e5 contributes 20 pairs
from 15 distinct query studies, PubMedBERT 20 pairs from 13; only **7 query studies
appear in both** tails. A pair present in one tail and absent from the other is not
evidence that the other candidate handled it — it may sit at rank 60 and simply not
be among that candidate's twenty worst.

Group labels are also candidate-specific. Length and heading groups are defined by
*that candidate's* tokenizer, so the same study can fall in different groups:

| Query study | e5 tokens / groups | PubMedBERT tokens / groups |
| --- | --- | --- |
| Study A | 573; over-512; lost-heading | 489; within-512; retained-heading |
| Study B | 388; within-512; no-heading | 308; within-512; no-heading |
| Study C | 308; within-512; no-heading | 252; within-512; no-heading |
| Study D | 277; within-512; retained-heading | 216; within-512; retained-heading |
| Study E | 160; within-512; retained-heading | 125; within-512; retained-heading |
| Study F | 142; within-512; retained-heading | 115; within-512; retained-heading |
| Study G | 105; within-512; retained-heading | 86; within-512; retained-heading |

Across these seven shared studies the e5 tokenizer produces roughly 17–28% more
tokens than PubMedBERT's on the same report. That is the mechanical reason the
candidate over-512 groups differ so much in size — 45 studies for e5, 19 for
PubMedBERT — and it is why the two candidates' subgroup rows in the results table
must not be read as a head-to-head on "long reports".

Two subgroup cells in particular should not be quoted without their denominators:
PubMedBERT's **0.0000% over-512 R@1 is 0 of 19 studies**, and its **−25.0000 pp
lost-heading row is 4 studies** — one study held the group's entire baseline credit.
Neither is a stable estimate.

## Tail composition

Recomputed from the packet's forty pair headings.

| | e5-base-v2 | pubmedbert-base-embeddings |
| --- | --- | --- |
| image_to_text / text_to_image | 14 / 6 | 7 / 13 |
| source-English / translated | 9 / 11 | 12 / 8 |
| within-512 / over-512 | 18 / 2 | 18 / 2 |
| retained / no / lost heading | 9 / 9 / 2 | 14 / 6 / 0 |
| distinct query studies | 15 | 13 |
| pairs worse than baseline rank | 13 | 9 |
| pairs better than baseline rank | 7 | 11 |

Three things in that table are worth stating and one is worth discounting.

**Provenance.** Validation is 175 source-English and 266 translated (39.7% / 60.3%).
PubMedBERT's worst twenty are 60% source-English — an overrepresentation relative to
base rate — while e5's are 45%. This is directionally consistent with the aggregate
subgroup rows, where both candidates lose on source-English (−5.14 pp for e5, −4.29
pp for PubMedBERT) and both post small translated-group gains (+0.38 and +0.75 pp).
Twenty pairs cannot establish the size of that asymmetry, and the translated "gain"
is a fraction of a point in a group of 266.

**Where the error sits relative to baseline.** e5's tail is mostly *regression*: 13
of 20 pairs rank worse than the baseline did, and 10 pairs collapse from a baseline
rank of 50 or better to 100 or worse — pair 6 goes from rank 2 to 215, pair 12 from 7
to 155, pair 3 from 40 to 299. PubMedBERT's tail is mostly the opposite: 11 of its 20
pairs *improved* on the baseline rank and are still bad (pair 3, 305→176; pair 6,
230→147; pair 14, 199→102), with only 2 collapses from a baseline rank under 50. The
two candidates lose similar aggregate ground by different routes — e5 by breaking
queries the baseline handled, PubMedBERT by failing to rescue queries the baseline
already handled badly.

That split is partly mechanical and must not be read as a like-for-like count. Each
set is *that candidate's* own twenty worst by its own rank, so the two tails sample
different populations: a candidate whose errors are deeper fills its top twenty with
different material than one whose errors are shallower. The same caution that applies
to the tokenizer-defined group labels above applies here.

**Direction mix is only weakly informative.** e5's tail skews `image_to_text` (14/6)
and PubMedBERT's skews `text_to_image` (13/7). For e5 this matches its directional
mean ranks (i2t 19.97 vs t2i 17.33 — a heavier i2t tail) even though its i2t R@1 is
the *higher* of the two (9.2971 vs 8.3900); for PubMedBERT the t2i-heavy tail is not
matched by any mean-rank difference at all (15.74 vs 15.88). Since the skews point
different ways relative to each candidate's own directional metrics, the mix should
not be read as a property of either encoder.

## What the reports actually show

### Retrieval of normal or near-normal reports for abnormal queries

The most frequent readable pattern in both tails. A query describing specific
positive findings retrieves a report that denies them.

- e5 pair 1: a translated report describing a pes anserinus exostosis, a PCL partial
  tear, grade II/III chondromalacia and patellar dislocation retrieves a fully
  templated normal right-knee report.
- e5 pair 7: post-ACL-rupture with a complex medial meniscal tear and an
  osteochondral lesion retrieves a report of unremarkable morphology and signal
  intensity in the same house style.
- e5 pair 14: ACL thinning, lateral meniscus amputation and patellar tendinosis
  retrieves a left-knee report of no internal derangement.
- e5 pair 17: a multiloculated ACL-origin ganglion with heterogeneous marrow
  retrieves an unremarkable examination — sharing the query's own sentence frame
  (an "otherwise normal" list of the cruciate ligaments and other structures).
- e5 pairs 19 and 20 and PubMedBERT pairs 4, 5 and 20 follow the same shape.
- PubMedBERT pair 9 is the near-normal variant: a query describing grade IV medial
  chondromalacia, extensive condylar marrow edema and a vertical posterior-horn tear
  retrieves a report whose menisci are unremarkable and whose worst finding is a
  grade 3 patellar facet fissure.

Pair 17 is the clearest case of format similarity dominating finding content: query
and retrieved report are plainly from the same dictation template, and the retrieved
one is the version with nothing wrong in it.

### Shared templates with opposed findings

Where query and retrieved share a structured house template, the retrieved report is
often the template's *other* extreme.

- e5 pair 3: an acute pivot-shift injury — complete ACL tear, lateral meniscal tear,
  MCL and FCL sprains — retrieves the identical `OSSEOUS STRUCTURES / JOINT SPACE /
  … / IMPRESSION` template reporting a normal ACL, normal menisci and isolated
  lateral chondrosis.
- e5 pair 13 and PubMedBERT pair 7 retrieve the same study (Study H) for the same
  query (Study I): a query of early undersurface tearing of the medial meniscal body
  retrieves a report of complex tearing of the medial meniscal posterior body and horn
  in a near-identical short template. Same structure, same compartment, different
  severity — both candidates make this error.
- PubMedBERT pairs 15 and 17: two long structured reports both retrieve the same
  medial-compartment study (Study J). In pair 15 the query is itself a medial
  root-tear report, so template and compartment agree and only severity and the
  fracture differ; in pair 17 the query is a *patellofemoral maltracking* report with
  an intact medial meniscus, and the retrieved report is medial-dominant throughout.

### Hub / attractor images in text_to_image

PubMedBERT's t2i tail repeatedly lands on a small number of the same images. Study
Study J is retrieved for two different queries (pairs 15, 17) and study Study E
for two more (pairs 18, 19) — 4 of its 13 t2i pairs land on just two studies. Study
Study E is simultaneously a *query* in PubMedBERT pair 16 (rank 92) and in e5 pair
16 (rank 130). e5 shows a milder version: Study K is retrieved for two different
queries (pairs 12, 15).

Notably, in PubMedBERT pair 19 the query is the acute ACL-tear / lateral-meniscal-tear
study and the retrieved report is a chronic medial-compartment degeneration report —
the hub is attracting queries whose findings it does not share.

### Laterality

Checked explicitly, and the result is the opposite of what the image-adaptation
review found. In these forty pairs, **wherever both the query report and the
retrieved report state an examination side, they agree** — e5 pairs 3, 7, 14 and 18,
PubMedBERT pairs 3, 12, 13, 15 and 17. Not one tail pair contradicts laterality.
(e5 pair 5 reads as an exception only on a word search: the query is a right-knee
exam and the retrieved report says "bilaterally" while describing the distribution
of degenerative change, not the side examined.)

What is common instead is *silence*. In 9 e5 pairs and 8 PubMedBERT pairs exactly one
of the two reports names a side, and in 6 and 7 pairs respectively neither does — many
translated reports and short house-style reports never state laterality at all. So in
roughly three quarters of these pairs laterality is simply unavailable as a
discriminator in the text.

This says the tail errors are not laterality confusions. It does not say the encoders
represent laterality correctly, because the worst pairs are the wrong place to look
for that: a pair that got the side wrong and was otherwise well matched need not rank
badly enough to appear here. It also does not bear on the two quarantined studies,
whose conflicting left/right reports are excluded from this cohort.

### Truncation, in the one place it is visible

Query study Study A is the only study in either tail whose truncation status
differs between candidates, and it is instructive precisely because both candidates
saw the same report.

- Under the e5 tokenizer it is **573 tokens, over-512, lost-heading**: the
  `IMPRESSION` block carrying the complete acute ACL tear, the lateral meniscal tear
  and both sprains falls past the cap. It appears in the e5 tail **twice** (pairs 3
  and 5) at ranks 299 and 233 against baseline ranks 40 and 41.
- Under the PubMedBERT tokenizer the same report is **489 tokens, within-512,
  retained-heading**, and it appears once (pair 19) at rank 90 against baseline 40.

Both candidates degrade on this study; the one whose tokenizer keeps the impression
degrades far less. That is a single study, post-selection, with no matched control,
so it is an observation consistent with the truncation concern rather than evidence
for it. It is, however, a concrete demonstration that the 512-token cap bites at
different places for different tokenizers on identical text.

### A study both candidates and the baseline fail

Query Study B — the translated pes anserinus exostosis report — is the worst pair
in *both* tails: e5 ranks its positive 411 (i2t) and 401 (t2i); PubMedBERT 420 and
406; the baseline 347 and 337. Out of 441 candidates, all three encoders place the
true pair near the bottom. Whatever is wrong here is not a property of the text
encoder under test. The report is long, translated, heading-free, and its findings
(a 1 cm stalked osseous projection, marrow reconversion, a small lateral condyle
edema focus) are the kind of small, focal, non-compartmental detail that three
sampled slices per plane may not contain at all. That is a hypothesis, not a finding;
confirming it would require looking at the images, which this review does not do.

### Data-quality observations, recorded not acted on

- e5 pair 10 retrieves a report whose entire findings section is the string `aa`. A content-free report outranks the true
  positive at rank 159. This is a cohort-content observation; no row is being removed
  and no label is being corrected here.
- PubMedBERT pair 6's retrieved report contains a mojibake artifact
  ("posterior horn**â€“**root junction"), i.e. a UTF-8 sequence decoded as Latin-1
  somewhere upstream of the packet. **Investigated 2026-09-11: it is in the supplied
  source data**, present in 14 of 4,407 cohort records, all source-English, none
  translated. See the follow-up section below.
- The retrieved report in PubMedBERT pair 15/17 (Study J) contains an ungrammatical
  sentence on broad-based grade 4 chondrosis, repeated verbatim in its impression —
  a dictation artifact in the source, not a translation artifact.

## Findings

1. Neither candidate qualifies for promotion, and this review does not change that.
   Both aggregate differences are negative with intervals spanning zero at one seed.
2. The two candidates fail differently. e5 regresses on queries the baseline handled
   well (10 of 20 tail pairs collapse from a baseline rank ≤50); PubMedBERT mostly
   fails to fix queries the baseline already got wrong (11 of 20 tail pairs improved
   on baseline rank and remain in the worst twenty). PubMedBERT's smaller aggregate
   loss is consistent with that difference, at one seed.
3. The dominant readable error in both tails is **template and register matching over
   finding content**: abnormal queries retrieve normal reports in the same house
   style, and shared-template pairs retrieve the opposite severity in the same
   compartment. The [image-adaptation review](claude-translated-dfd07db-image-block-error-review.md)
   describes the same pattern in its arm, so it is not introduced by swapping the text
   encoder. The overlap between the two reviews is only partial, though: that review
   also found laterality switched in two pairs, and laterality is never contradicted
   in these forty. Two signals, one shared and one divergent, are not enough to
   conclude the failure mode is encoder-independent.
4. Tokenizer-defined subgroups are not comparable across candidates. The over-512 and
   lost-heading rows for the two candidates describe different sets of studies
   (45 vs 19, and 22 vs 4), and the one study observable in both states degrades much
   more in the truncated state. Report the denominators whenever those rows are cited.
5. Retrieval-hub behavior exists in PubMedBERT's text_to_image tail (two studies
   absorb 4 of 13 pairs). **Resolved after this review was written — see the
   follow-up section below.** It is a general property of every run measured,
   including the baseline, and PubMedBERT is in fact the *least* concentrated of the
   three in that direction.
6. Nothing here attributes an error to image representation, text representation,
   translation, or source pairing, and no retrieved pair is treated as a clinically
   equivalent positive.

## Follow-up investigations — 2026-09-11, after the review was written

Two of the open threads above were closed from saved artifacts. Neither required a
model run: no checkpoint was loaded, no evaluation was launched, the test split was not
touched, and no protocol, cohort or selection decision changed. Reproduce both with
`.venv/bin/python scripts/analyze_text_screen_followups.py`.

### The mojibake is in the supplied source data, not in the pipeline

The `posterior horn<mangled>root junction` artifact in PubMedBERT pair 6 is the three
codepoints U+00E2 U+20AC U+201C — the UTF-8 bytes `E2 80 93` for an en dash, decoded
once as cp1252. Every file in the chain is itself valid UTF-8, so this has to be found
by codepoint; a terminal can re-mangle the glyphs in either direction.

It is already present in `data/sample/train.csv`: **13** mangled en dashes and **1**
mangled apostrophe against **822** clean en dashes and 259 clean apostrophes in the same
file. Sporadic and per-report. That file is the supplied dataset, not a project export —
it entered in the initial commit (`git log -- data/sample/train.csv` shows `9435b20`
only), nothing under `scripts/` or `src/` writes it (it appears solely as the
`reports_csv` *input* default in `config.py` and `build_contrastive_manifest.py`), and
the EDA notebook describes it as the provided study metadata and reports. So the damage
is upstream of this project. In the cohort manifest it affects **14 of 4,407** records (10 train, 2
validation, 2 test); all fourteen are `report_language: en` / `report_provenance:
source_english`, and `report` equals `source_report` byte-for-byte in all of them — **no
translated report is affected**, so the Claude translation path did not cause it. The
error-review packet carries one mangled sequence beside 14 clean en dashes, which is
exactly what faithful rendering of the stored text looks like; the packet builder is not
the cause. Corroborating: within this one report the Findings line is mangled while its
own Conclusion reads `horn-root` with a plain ASCII hyphen.

**Not repaired, and repairing it is not a typo fix.** The mangled text is inside
`report_sha256` / `source_report_sha256` — the stored hash reproduces exactly from the
stored string, and repairing the three characters changes it. That would invalidate the
cohort manifest SHA-256 (`8f6f021954e2…`) every completed run is pinned to. It affects
1 of 441 validation studies and remains an observation, not an action.

### Retrieval hubs are universal; hub identity is encoder-specific

`rankings-validation-best.jsonl` in each run directory records, per query, the rank-1
retrieval (`top_uid`) and the true positive's rank — 441 validation queries in each
direction. That is the whole pool, so finding 5 was answerable without the fresh
evaluation it assumed. The three runs share an identical 441-study candidate pool, and
the packet's twenty PubMedBERT pairs match the file on rank and retrieved UID 20/20.

| Run | t2i distinct rank-1 targets | biggest hub | top two | i2t distinct | biggest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline multilingual-e5 | 181 | 21 (4.8%) | 31 (7.0%) | 227 | 9 (2.0%) |
| e5-base-v2 | 143 | 26 (5.9%) | 45 (10.2%) | 193 | 12 (2.7%) |
| PubMedBERT | 179 | 11 (2.5%) | 20 (4.5%) | 215 | 9 (2.0%) |

Monte-Carlo null, 441 queries each drawing a rank-1 target uniformly at random over
20,000 draws: biggest hub p99 = 7, top-two p99 = 13, distinct targets p05 = 268. All
three runs sit far outside that null on every statistic, and `image_to_text` is milder
than `text_to_image` in all three. Concentration is therefore a general property of
these embedding spaces — not a tail artifact, and not introduced by swapping the text
encoder.

Read the table carefully rather than by one number: **e5 is the most concentrated of
the three on every statistic**; baseline and PubMedBERT are effectively tied on
distinct-target count (181 vs 179) and the *baseline* carries the heavier single hub
(21 queries vs 11) and heavier top two (31 vs 20). So PubMedBERT is not uniquely
hub-prone in `text_to_image` — on the single-hub measures it is the mildest of the
three, and on target diversity it is indistinguishable from the baseline. Whatever the
review saw is not a PubMedBERT-specific defect.

Why the tail view pointed at PubMedBERT specifically is *not* established. The worst
twenty are somewhat more hub-dense than a random twenty from the same run, but weakly:
distinct rank-1 targets among the worst twenty are 16 (baseline), 15 (e5) and 17
(PubMedBERT) against random-twenty means of 18.57, 17.66 and 18.79, with
P(random at least as concentrated) = 0.047, 0.068 and 0.111 over 20,000 resamples.
Directionally consistent in all three runs and consistent with the tail
over-representing hubs, but at n=20, uncorrected, that is not a settled effect.

Two corrections to the hub section above:

- The e5 case is not "a milder version". Study Study K is e5's single biggest hub
  in the run — rank-1 for **26 of 441** t2i queries, more than either PubMedBERT study.
  Only two of those 26 fell inside the worst twenty, which is all that section could see.
- PubMedBERT's Study J and Study E are genuine hubs but not its largest: 6
  queries each, the #12 and #16 hubs of 179. Its biggest is Study L at 11.

Hub *identity*, unlike hub behavior, does not transfer. Top-10 hub sets share 3/10
between baseline and PubMedBERT in `text_to_image`, 1/10 for the other two pairings and
**0/10 across all three**; in `image_to_text` every pairing shares **0/10**. Each
encoder builds its own attractors. The biggest hubs are also not one recognisable type —
a 2,293-character templated source-English report, a 636-character translated report, a
595-character terse all-normal report and a 1,636-character abnormal report, spanning
both provenances and both lateralities. That is compatible with the template-and-register
reading in finding 3, but this measurement does not establish it.

**Limit:** this is rank-1 concentration over the full pool, not each query's complete
441-long ordering. The orderings were never saved and recovering them would need a fresh
evaluation pass, which was not authorized and was not run.

## Recommendation for the next experiment — proposal only

Matrix selection is a one-way door (`experiments.py`, commit `298dec1`) and no launch
is authorized by this document. The following is put to the user for a decision.

**Preferred: text-only final-block adaptation on the baseline encoder, full cohort,
three matched seeds (2026 / 2027 / 2028).**

The argument that carries this is that it is the untested half of the adaptation
question: image-only final-block adaptation is complete at three matched seeds with a
mean gain of +1.2094 pp, and the text side has never been tried on the full cohort.
It also holds the encoder fixed at multilingual E5, so no tokenizer-defined subgroup
shifts underneath the comparison — which, after finding 4, is worth something on its
own. Finding 3 is a secondary and weaker reason: register-over-findings matching is
the sort of thing a trainable final text block could plausibly move, but that
observation is descriptive and predicts neither the direction nor the size of any
gain. Budget three seeds up front; single-seed screens have now twice returned
intervals spanning zero.

Two alternatives worth considering, both of which this review supports less well:

- **Chunked full reports.** Finding 4 shows truncation bites unevenly, but exactly one
  study across the forty pairs is observable in both truncation states, and chunk
  aggregation would need a new representation contract — a large change on thin
  evidence.
- **A hard-negative or template-aware objective.** Finding 3 is the most frequent
  pattern in this review, but changing the objective breaks comparability with every
  completed arm. Not advisable before the adaptation question is settled.

Deliberately *not* recommended: another single-seed frozen-encoder screen. Two have
now returned negative point estimates with intervals including zero, and a third
would not resolve the encoder question either.

Unchanged by this document: the cohort and quarantine, the single-positive objective
with uniform expected tie credit, the frozen-matrix selection rule, and the locked
test set. Multilingual E5 remains the reference; final model selection remains open.
