# What the series CSV cannot tell you, verified on this cohort

Measured 2026-09-14 (Asia/Taipei). Prompted by [`prev/rsna_knee_101.ipynb`](../prev/rsna_knee_101.ipynb),
a synthesis of public RSNA-knee work by earlier teams. **Every claim below was re-verified against
this project's own data rather than taken from the notebook.**

Evidence: [`crop-retention.json`](crop-retention.json), [`fov-and-depth-audit.json`](fov-and-depth-audit.json).
Scripts: [`scripts/audit_fov_and_depth.py`](../scripts/audit_fov_and_depth.py),
[`measure_crop_retention.py`](../scripts/measure_crop_retention.py),
[`check_slice_grouping.py`](../scripts/check_slice_grouping.py).

## 1. `Fluid_Sensitive` and `Fat_Suppression` are one bit, not two

**Verified here: identical on 24,371 of 24,371 rows — 100.0000%.** They name physically independent
properties (fluid sensitivity is set by TR/TE weighting; fat suppression is a preparation applied on
top), so four combinations are possible and the table records one.

**This project is affected.** `pick_study_slots` (`data.py:161`) documents itself as preferring
*"fluid-sensitive series if any exist for the plane"*. Because the column is a byte-copy of
`Fat_Suppression`, **it is in fact selecting fat-suppressed series**, and the stated preference was
never implemented. Every cohort in this project was built through that selection. Fat-sat PD is a
defensible default, so this is not necessarily a bad choice — but the docstring asserts something the
data does not support, and any successor reasoning from it will be wrong.

**Recovering the real axes requires the DICOM header** — `RepetitionTime`, `EchoTime`, `ScanOptions`,
`SeriesDescription` — not the CSV. Two string-matching traps, both of which *silently invert* the
answer: underscore is a word character, so a token test for `we` never fires inside `t2_de3d_we_tra`;
and `ScanOptions` must be matched as exact tokens, because one vendor writes `SAT_GEMS` for *spatial*
saturation and a substring test for `SAT` marks non-suppressed series as suppressed.

## 2. Filename order is not anatomical order — and this pipeline already handles it

Filenames are SOP Instance UIDs, i.e. random. **Verified here: 0 of 180 series have filename order
equal to anatomical order in either direction, and only 6.9% of filename-adjacent pairs are
anatomically adjacent.**

**`ordered_slice_paths` (`preprocessing.py:150`) is correct** — geometric position first, then
`SliceLocation`, then instance number, then UID, each fallback requiring the field on *every* slice.
No change needed.

**One audit script written this session did have the bug** and was fixed: `measure_crop_retention.py`
decoded `files[len // 2]` under filename order, i.e. an arbitrary slice. Re-measured with
`ordered_slice_paths`, **every retention figure moved by ≤1pp** and no decision changed — knee
cross-sectional width is stable along the stack. The other two scripts compute gaps from *rank*
arithmetic (mean index spacing × mm per slice), which is unaffected by what sits at each rank.

## 3. Field of view, and why 130mm was stuck

True FOV, from the DICOM (`series_fov_mm`, new): **median 160mm**, p10 150, p90 180, range 130–320.
`ReconstructionDiameter` is **absent from 100%** of series, so FOV must be derived as
`PixelSpacing × Rows`.

Tissue retained by a centred crop, Otsu foreground mass, **n=1200 series, anatomically central slice**:

| crop mm | mean kept | p10 | ≥95% kept | mm/px at 224² |
| --- | --- | --- | --- | --- |
| 130 (previous) | 0.861 | 0.706 | 32% | 0.58 |
| 140 | 0.915 | 0.803 | 44% | 0.63 |
| **150 (chosen)** | **0.957** | **0.884** | **67%** | **0.67** |
| 160 | 0.984 | 0.943 | 89% | 0.71 |
| 180 | 0.998 | 1.000 | 99% | 0.80 |

**The old guard could not be raised.** `if 16 < wanted < min(height, width)` reduces to
`FOV > crop_mm`, so any constant above the cohort minimum of 130mm would have been **silently skipped**
for some series, leaving them at a different mm/pixel. Cropping alone cannot normalise scale.
`_centre_to_extent` now crops **or zero-pads** to an exact extent. The earlier team reached the same
conclusion independently and chose 160mm; this project chose **150mm** to keep padding rarer and
preserve resolution for small findings.

**Read the retention numbers as an upper bound.** Otsu foreground counts any tissue, including thigh
and calf muscle, not "diagnostically relevant joint".

## 4. Depth sampling discards most of the volume

- **89.9%** of slices are never read; median **31.3mm** between sampled slices.
- The three channels of one input straddle a median **62.7mm** — wider than the joint. DINOv2 is
  being fed three anatomically unrelated images stacked as RGB.
- `slice_group_indices` (new) with K=5 contiguous groups: coverage 10% → **50%**, channel spread
  62.7mm → **7.0mm**. Cost is linear, 3 → 15 encoded views per study.

## 5. Studies carry far more data than the pipeline opens

Median **5 series per study** (mean 5.67, max 11); **100% of studies have more than the 3 used**.
The unused series are **not localizers** — 0 of 202 sampled: T1 34.7%, PD 19.8%, T2 17.8%,
unlabelled 13.4%, other 13.4%, median ~30 slices each. **The pipeline reads 9 of ~194 slices per
study = 4.6%.**

## 6. The trap to clear before using more series

Protocol tracks indication: a knee scanned with an extra sequence was scanned by someone looking for
something, so **series count may leak label information** — which would re-inflate exactly the probe
figures the gold-holdout work deflated.

**Checked 2026-09-14** ([`protocol-leak-check.json`](protocol-leak-check.json),
[`scripts/check_protocol_leak.py`](../scripts/check_protocol_leak.py)). Series counts read from
**disk**, not from the CSV. Median 5 per study, range **3–14**, across all 4,407 studies.

**Primary — series count → each gold label, n=58, permutation test.** **0 of 12 labels flagged at
p<0.05** (≈0.6 expected by chance alone); mean |AUROC−0.5| = **0.058**, largest PF OA at 0.593
(p=0.227). No label separates.

**State the power honestly, because 58 studies is not many.** The permutation null spans
[0.351, 0.648] for the best-powered label (Effusion, 35 positives) and [0.305, 0.697] for the worst
(MCL, 9). So this check can only detect **|AUROC−0.5| > 0.149–0.197**. It rules out *large* protocol
leakage; it **cannot** rule out a modest one around AUROC 0.60.

**Proxy — series count vs report length, n=4,407.** Spearman **−0.032** (p=0.040). Significant only
because n is large — it explains ~0.1% of variance — and the sign is **negative**, i.e. richer
protocols go with *shorter* reports, the opposite of the leak hypothesis.

**Verdict: proceed with variable-length slots, with a residual risk that is bounded but not zero.**
Recommended mitigation, since it is nearly free: record series count per study and include it as a
control when reading any probe result, so a modest leak would be visible rather than silent.

## 7. The recovered axes, measured on all 24,371 series

`src/contrastive_learning/acquisition.py` (new) recovers both axes from the header, with the two
inverting string traps guarded and unit-tested: `t2_de3d_we_tra` **is** suppressed (underscore is a
word character, so separators must be normalised before `\bwe\b` can fire), and `SAT_GEMS` **is not**
(GE's *spatial* saturation; `ScanOptions` is matched token-wise, never as a substring).

Full pass ([`acquisition-slots.json`](acquisition-slots.json),
[`scripts/audit_acquisition_slots.py`](../scripts/audit_acquisition_slots.py)):
**PD 13,646, T1 5,383, T2 5,114, GRE 226, UNK 2.** Weighting is decided by the description for
18,777 series and by the **TR/TE fallback for 5,594 (23%)** — so roughly a quarter of the corpus
rests on the timing rule, which is worth remembering before trusting it as ground truth.

### The pipeline picks the right series for the wrong reason

| | fat-sat | fluid-sensitive | weightings |
| --- | --- | --- | --- |
| **selected** by `pick_study_slots` | 95.8% | **99.9%** | PD 10,730, T2 2,479, T1 10 |
| **discarded** | 11.4% | 49.8% | **T1 5,373**, PD 2,916, T2 2,635, GRE 226 |

The CSV bug in §1 is real, but fat-suppression and fluid-sensitivity are tightly correlated in this
corpus, so the selection lands on fluid-sensitive fat-sat PD anyway — **99.9% fluid-sensitive**.
`pick_study_slots` gets the right answer for the wrong reason. **No cohort needs rebuilding**, and
nobody should rely on that coincidence. What it discards is dominated by **T1 non-fat-sat**, the
complementary contrast the model has never seen.

### Slot fill, which is what should define the slots

| fill | slot |
| --- | --- |
| 84.0% | Coronal / PD / fatsat |
| 82.6% | Sagittal / PD / fatsat |
| 72.2% | Axial / PD / fatsat |
| **64.1%** | **Coronal / T1 / no-fatsat** |
| 41.1% | Sagittal / T1 / no-fatsat |
| 37.5% | Sagittal / PD / no-fatsat |

**Only 2 of 21 slots clear 80%.** Define slots from the top of this list, not from the product of
every flag — a slot present in 20% of studies is mostly a masked-out zero.
**`Coronal / T1 / no-fatsat` is the best single addition**: the highest-fill slot carrying contrast
the pipeline never opens.

## 8. Three more checklist items, verified here

- **Rescale slope/intercept: handled** (`_decode_pixels`). Not cosmetic — 201 of 1,200 sampled series
  carry a non-identity slope (e.g. 4.78), so skipping it would distort percentile normalisation for
  ~17% of the corpus.
- **MONOCHROME1: not handled, and currently harmless.** `_decode_pixels` never reads
  `PhotometricInterpretation`, so a MONOCHROME1 series would be silently inverted. **0 of 1,200
  sampled series are MONOCHROME1** (all MONOCHROME2), so nothing is wrong today. It is a latent trap
  if this pipeline ever meets another source; a two-line guard would close it.
- **Laterality: resolved for 98.3%** of studies (R 2,298, L 2,034, unresolved 75 = **1.70%**), which
  are left unmirrored rather than guessed. `resolve_laterality` implements the full prescribed chain
  — `Laterality` → `ImageLaterality` → sign of the geometric centre, with a 20mm midline guard. Four
  of the twelve targets are side-specific, so this mattering was not obvious in advance; it is fine.

## 9. A windowing trap that depth grouping would walk into

The checklist says window per **series**, not per slice, because per-slice normalisation rescales away
the brightness of an effusion on exactly the slices where it is the finding. `read_slot` takes its
1st/99th percentiles over the whole stacked slot, so it satisfies this **for the three slices it
currently samples**.

**But the window is computed over the sampled slices only, not the whole series.** Read K depth groups
as separate calls and each group gets **its own** window, so relative brightness across depth is lost
— the same failure one level up. **Any depth-grouping implementation must compute the intensity window
once per series and share it across all groups.** Recorded here because the code does not currently
prevent it.

## What is NOT transferable from the source notebook

That notebook targets a competition where **reports exist at fit time and not at predict time**, so it
rules out a text branch and uses reports only to manufacture targets. **This project's setting is
different** — reports are used to learn a representation, and the probe reads image features alone, so
a text tower is legitimate here. Do not import that constraint.
