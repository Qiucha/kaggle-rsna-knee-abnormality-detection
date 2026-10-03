# Protocol — does the 0.7105 gold-probe AUROC survive contact with unexposed data?

Pre-registered 2026-09-14, **before the first run**. Authorized by the user 2026-09-14.

## The question

Every published probe figure on this project is a **development statistic by construction**:
[the gold-case inventory](gold-case-inventory.md) established that 56 of the 58 gold studies were
image–report training pairs (48) or members of the checkpoint-selection split (8), and the other two
are in the locked test. The [matched-N audit](matched-n-loss-audit.md) then measured *how much* that
matters in principle — the encoder retrieves its own training pairs **3.0×** better than unseen ones
(train R@1 0.404 vs validation 0.134 at N=441).

Those two facts together say the 0.7105 headline **might** be substantially exposure-inflated. They
do not say by how much. Nothing in the repository does, because no unexposed gold data existed.

**This experiment creates some.**

## Design

Hold all 56 non-test gold studies out of **both** representation training and checkpoint selection,
retrain, and probe on those same 56 studies — now never seen by the encoder in any capacity.

- **Cohort:** `data/processed/contrastive-cohort-gold-holdout-v1.jsonl`, sha256
  `d104337696a19d52d12480c1cfaae30c937916102e36a67e3124755fa20123f5`. Derived by
  `digest/handoff/build_gold_holdout_cohort.py` from `contrastive-cohort.jsonl`
  (`54bd803e23acf...`) by moving the 56 studies to split `excluded` — a value the manifest contract
  already defines, and which `load_cohort(...).split("train"/"validation")` filters out by
  construction. Counts go 3,526/441 → **3,478/433**; the **440-study test split is byte-identical**
  and every other field on every row is unchanged (asserted in the builder).
- **Arms:** 4-block image unfreezing, three matched seeds 2026/2027/2028. Every other argument is
  copied verbatim from the queue that produced the published baselines
  `image-block-depth-v1/original-unfreeze4-seed{2026,2027,2028}`. **One lever: the cohort.**
- **Comparator:** those three published runs, already probed as the `after_cl` arm of
  `gold-probe-extended-v3` (macro AUROC **0.7105**).
- **Probe:** the *identical* specification — `LogisticRegression`, L2, `C=1.0`, `max_iter=5000`,
  `class_weight=null`, StandardScaler inside each training fold, `StratifiedKFold(8, shuffle=True)`
  with `fold_seed=2026`, pooled out-of-fold AUROC, bootstrap 5,000 resamples at `seed=2029`,
  `uniform_mean_768` features. **Nothing is tuned.** Labels from `data/cleaned.csv` — never
  `data/drop.csv`, which is the blank template, and the probe asserts a non-zero positive count.
- **Output root:** `checkpoints/contrastive/gold-holdout-v1/`, `experiment_version`
  `gold-holdout-v1`. New root and new version because `validation-selection.json` is a one-way door
  on every existing matrix.

## Estimand, fixed in advance

**Primary:** the **pooled paired difference** in macro out-of-fold AUROC, unexposed minus exposed,
across the three matched seeds — bootstrapped over **studies**, using the **same resample indices for
both arms** and resampling the *per-study difference*, not each arm independently.

This is pre-registered because the independent-CI route cannot work here: per-seed intervals on a
56-study, 12-label probe are ~±0.02–0.03, and comparing two such intervals by overlap is exactly the
underpowered rule that made the plane-pooling session record a null it could never have avoided. The
paired construction costs nothing — identical folds and identical resamples are already available —
and it is the rule that can detect an effect worth acting on.

**Secondary:** per-label paired differences with positive/negative counts beside each; MCL
(8 positives) reported pooled-OOF only and declared uninformative per fold in advance.

**Both outcomes are results, and both are pre-registered as such:**

- **Unexposed ≈ exposed** ⇒ exposure did not materially inflate 0.7105. Concern #3 is weaker than
  feared and the probe figures can stand with their existing caveat.
- **Unexposed materially below exposed** ⇒ the headline is substantially an artifact of exposure,
  and **every probe figure in `digest/` and in the HTML pages needs relabelling**, not just
  footnoting.

## Confounds, stated before the numbers

1. **The training set is 48 studies smaller** (3,526 → 3,478, −1.4%) and the selection split 8
   smaller (441 → 433). Dropping training data can only hurt, so the rule is asymmetric and fixed
   here: **if the unexposed arm lands at or above the exposed arm, no control is needed** — a null
   is already the conservative reading. **Only if it drops** is a control required, and then it is
   three more seeds dropping 48 *random non-gold* studies, to separate "lost 48 studies" from "lost
   the exposure."
2. **One residual text-exposure channel, measured: 1 study.** Exactly one study remaining in training
   carries a `report_sha256` identical to a held-out gold study's report, so the model still sees
   that report text paired with a different patient's images. One of 3,478. Reported either way;
   detected by hash, which is exact and threshold-free.
3. **Still not patient-level.** De-identification re-randomised every identity tag, so this is a
   study-level estimate. It is a **better** estimand than a development statistic, not a clean one.
4. **Still 56 studies.** Wide intervals are expected; the paired bootstrap is what makes the
   *difference* estimable, not the level.
5. **Validation R@1 is not comparable to the published 0.1383.** The new runs select over **433**
   candidates, not 441. Selection stays internally valid, but the two numbers must never share a
   table — the exact trap documented in [the matched-N audit](matched-n-loss-audit.md).

## Execution rules

Three seeds run **sequentially, into distinct output directories** — concurrent runs interleaving
into one `metrics.jsonl` cost a deleted directory in an earlier session. After the probe, **assert
the new arm key appears in the output**: `build_towers` silently skipped unrecognised arm keys until
that was fixed, and a complete-looking artifact with a missing arm is this project's characteristic
failure. The locked test split is not loaded at any point.
