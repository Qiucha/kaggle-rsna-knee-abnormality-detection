# Translation update assessment

Fetched origin/config at `dfd07db` (use the full authoritative commit in the
[JSON audit](translation-update-dfd07db-audit.json)); changes since `9bec1bd` are
`18f4197` (manual edits) and `dfd07db` (notes on non ASCII characters).
The remote CSV/notebook snapshots are under `checkpoints/translations/config-dfd07db/`.
Existing working branch, frozen cohorts, and completed checkpoints remain intact.

There are exactly five changed CSV rows, all confined to `LLM Translation`.
All 4,407 unique study IDs, source reports, and other CSV values are unchanged.

| CSV index (zero-based) | Split | Edit |
| --- | --- | --- |
| 814 | validation | Remove parenthetical German “Jägerhut” after “Hunter's cap” |
| 1607 | train | Expand chondromalacia grade notation |
| 2236 | train | Expand chondromalacia grade notation |
| 3159 | train | Expand chondromalacia grade notation |
| 4011 | train | Remove Turkish heading while retaining “LEFT KNEE MRI” |

The exact pinned E5 tokenizer, `query: ` prefix, full-report view and 512-token
limit produce different inputs for all five rows. None of these edits disappears
through truncation. Four affected training studies are 0.113% of training; the
single validation study is 0.227% of validation. This small scope does not prove
zero metric impact. The wording edits do not indicate a broad alignment failure;
no comprehensive clinical adjudication was performed.

The original translated baseline has already completed all 20 epochs in 29.03
minutes of measured training. Best checkpoint: epoch 16. Validation mean Recall@1:
9.7506% (image-to-text 10.2041%, text-to-image 9.2971%). Checkpoint reload reproduced
all stored metrics exactly. The earlier multilingual baseline was 10.66%; this
single run does not establish a significant difference. No test evaluation ran.

Decision: retain the completed `9bec1bd` run as the first translated baseline;
these edits do not require discarding or restarting it. No new training launched.
If future pipeline comparisons adopt `dfd07db`, run a separate fresh baseline on
that exact corpus first, keeping settings/splits fixed. The four changed training
inputs mean validation-only re-evaluation would not represent training on the
updated corpus, and resuming old weights would mix versions. Do not silently
replace the existing cohort or label its results as `dfd07db`.
