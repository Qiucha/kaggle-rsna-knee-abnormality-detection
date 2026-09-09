# Updated-translation controlled comparison

Status: completed / complete

Two confirmed conflicting duplicate-image training entries are excluded from both configurations. Validation remains 441 studies; test remains locked.

| Run | Seed | Best epoch | Mean validation R@1 |
| --- | ---: | ---: | ---: |
| full | 2026 | 19 | 10.6576% |
| impression_first | 2026 | 19 | 9.6372% |

Paired query differences (impression-first minus full), percentage points:

```json
[
  {
    "seed": 2026,
    "mean_difference_pp": -1.0204081632653061,
    "descriptive_paired_query_bootstrap_95_ci_pp": [
      -3.061224489795918,
      1.0204081632653061
    ]
  }
]
```

Intervals are descriptive: validation selected checkpoints; unknown repeated patients and template/report duplication remain limitations. These are exact-pair retrieval metrics, not clinical accuracy. Every attempted run is retained. No encoder unfreezing is launched by this bounded comparison.

## Review and decision — 2026-09-07

Retain full text; reject promotion of impression-first for the next controlled
experiment. Both selected epoch 19. The candidate loses 1.0204 percentage points
of mean R@1; the descriptive paired-study bootstrap interval is [-3.0612, 1.0204]
points. This does not establish a general ordering disadvantage. The predeclared
positive-gain rule did not trigger, so seeds 2027/2028 were not run; seed stability
is unmeasured. The queue completed at 06:01:04 Asia/Taipei without recovery.

Independent artifact checks verified 46 snapshot hashes, the quarantined manifest
hash, matched configs differing only in report view/output directory, contiguous
20-epoch histories, checkpoint selection, checkpoint hashes, zero reload deltas,
882 direction/query ranking rows per run, telemetry completion and paired results.
See [machine-readable audit](claude-translated-dfd07db-comparison-audit.json).
Reproduce with `.venv/bin/python checkpoints/translations/config-dfd07db/review_comparison.py`
using the cached tokenizer offline. This is a CPU artifact audit, not retraining.

Training took 25.40 minutes for full text and 25.53 for impression-first, excluding
reload. Both peaked at 1.808 GiB allocated GPU memory. Both log an infinite mean
gradient norm at epoch 2; later norms and selected evaluation metrics are finite.
Training uses AMP GradScaler, but the logs do not count skipped optimizer steps,
so their exact number cannot be recovered from these summaries. No learning-curve
or exact-reload failure was observed. Mean rank improves slightly in both directions
while R@1/5/10 decline; the declared R@1 selection metric remains authoritative.

### Long-report behavior

Groups use the original validation report's token length and the same 441-candidate
retrieval pool. Scores average both directions; subgroup memberships overlap.

| Validation group | Studies | Full R@1 | Impression-first R@1 | Difference, pp |
| --- | ---: | ---: | ---: | ---: |
| Over 512 tokens | 67 | 9.7015% | 7.4627% | -2.2388 |
| At most 512 tokens | 374 | 10.8289% | 10.0267% | -0.8021 |
| Changed model input | 254 | 10.2362% | 8.8583% | -1.3780 |
| Unchanged model input | 187 | 11.2299% | 10.6952% | -0.5348 |
| Late summary heading in original | 43 | 6.9767% | 4.6512% | -2.3256 |

The candidate gains 34 and loses 43 direction/query top-1 successes overall.
Recovering 42/43 late headings did not yield an observed subgroup gain. These
small, post-selection subgroups are descriptive. Changed training representations
and competing report embeddings also affect unchanged-input queries; these are
not isolated causal effects of moving a given query's heading.

### Candidate worst-pair report review

Read all 20 worst candidate direction/query pairs, sorted by expected rank
(descending), direction and UID. The local packet includes full paired reports,
UIDs, baseline ranks and candidate scores at
`checkpoints/translations/config-dfd07db/candidate-error-review-packet.json`.
Five entries (four unique query studies) exceed 512 tokens. This review compares
report statements; the candidate image's paired report is a proxy in text-to-image
retrieval. It is not independent image interpretation or clinical adjudication.

| Packet entries | Provisional report-level observation |
| --- | --- |
| 1, 2 | The tibial/pes osseous projection is absent from both competitors; one describes a large nerve-sheath ganglion and the other different ACL/PCL findings. Both queries are short. |
| 3, 7 | Complex medial meniscal, partial ACL and MCL findings compete with acute ACL injury without meniscal tears, or degenerative disease without ACL tear. |
| 4, 6 | The query's findings describe medial meniscal tearing, but its conclusion says no other significant abnormality. Competitors deny meniscal tearing. Reordering preserves this source inconsistency. |
| 5, 9 | Early medial meniscal body tearing competes with normal menisci or a complex posterior-horn tear. |
| 8, 12 | Medial meniscal flap/extrusion and cartilage damage compete with normal menisci/cartilage, or partially overlapping posterior-horn/root disease of different morphology. |
| 10 | Advanced patellofemoral cartilage damage competes with inflammatory synovitis and explicitly normal cartilage. |
| 11, 17 | Short ACL/medial meniscus injury query competes with Hoffa impingement/Baker cyst or overlapping meniscal disease without the reported ACL injury. |
| 13 | Similar long template and medial degeneration; competitor adds subchondral stress fracture while query describes no fracture. |
| 14, 18 | Acute ACL/lateral meniscal injury competes with intact ACL/no meniscal tear or advanced degeneration with ACL continuity. Candidate ranks worsen from 40/41 to 97/89. |
| 15 | Recent patellar dislocation with unremarkable menisci competes with lateral meniscal tearing, cartilage loss and quadriceps injury. |
| 16 | Normal menisci and focal cartilage thinning compete with a bilateral combined report containing degenerative meniscal tears, insufficiency fracture and substantial effusion. |
| 19 | Chondroid lesion, maltracking and chondrosis without meniscal tear compete with medial meniscal root tearing and no focal cartilage defects. |
| 20 | Chondrosis without meniscal tear competes with ACL/meniscal injury. Competitor also contains a medial/lateral meniscus wording inconsistency between section text and impression. |

Some severe ranks improve without producing top-1 success. Short-query errors and
persistent report disagreements argue against treating truncation as the sole
remaining problem. No source reports, positives, or labels were rewritten.

### Next selected experiment

Compare image-only final-block adaptation against the retained quarantined
full-text baseline: same manifest, split, full report, multilingual E5 mean pooling,
512 tokens, batch 32, seed 2026, 20 epochs, model revisions and checkpoint rule.
Change only image unfreezing from 0 to 1; keep text frozen and the existing image
learning rate 1e-5. Start from the pinned pretrained encoders, matching baseline
initialization rather than continuing its trained projection checkpoint.

Before running, freeze a separate execution snapshot/config and pass a GPU smoke
with reload. If seed-2026 mean R@1 beats the matched baseline, repeat baseline and
candidate at seeds 2027/2028; otherwise stop this comparison. Review image-only
results before text-only adaptation. Both-block adaptation remains conditional.
This review selects the next experiment; it has not launched it. Test stays locked.
