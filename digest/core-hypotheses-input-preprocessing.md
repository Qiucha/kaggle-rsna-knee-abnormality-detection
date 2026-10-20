# Core hypotheses: input shape, 3D vs 2.5D, and preprocessing

Status: **final (v6)**, 2026-09-27. Adds rounds 5–7 (in-plane levers, ROI views, confirmation, codec review) to the
v5 study (rounds 1–4). Every round was followed by an adversarial critic.
- Raw reports, code and logs: `.scratch/input-preproc-hypotheses/` (`r1-*` … `r7-*`, `*-critique.md`,
  `experiments-log.md`, `code/`).
- Experiment outputs: `checkpoints/input-preproc-hypotheses/`.
- **Next-step tickets: `.scratch/image-preproc-next/`** (`spec.md` + `issues/01–08`).
- No tracked file was edited except this note, and nothing was submitted to Kaggle. One private Kaggle kernel (decode timing,
  no submission) was run by a peer session (`r4b-kaggle-decode.md`).

## v6 update (rounds 5–7): what changed

- **Every inference-only input change tested fails.** Still untested: a properly localised knee-centred crop/ROI (H2.3, H3.1),
  and mm-anchored slices (H1.3b). On the deployed encoder (A), none of these passes the gate:
  - antialias: ±0.001;
  - Otsu-centroid crop: +0.007, gold −0.006;
  - fixed zoom 100/115/150 mm;
  - 336 px: +0.008 [+0.001, +0.016] LM-test, but only +0.006 vs deployed;
  - an image-centred 130 + 90 + 65 mm zoom pyramid: +0.004, and ≈ its off-centre view-count control;
  - pooled patch tokens in the zoom boxes: −0.008, Holm-significant loss.

  On frozen DINOv2 (U) nothing met a pre-registered rule either. All arms: `r5-inplane.md`, `r6-roi.md`, `r7-confirm.md`.
- **The biggest pipeline risk is decoding, not preprocessing.**
  - The Kaggle offline image has **no JPEG-Lossless or JPEG-LS decoder** (r4b), and the hidden test contains JPEG Lossless.
  - A JPEG-LL slice is silently replaced by its nearest decoded neighbour. A fully JPEG-LL plane crashes the submission.
  - Simulated cost of substitution: ≈ −0.005 LM-test macro per 10% of planes affected.
  - This is a BLOCKER for any change to series or slice selection.
  - A vendored-wheel fix (pylibjpeg + libjpeg + openjpeg, 5.4 MiB) is prototyped and passes a bit-exact fixture self-test
    on CPython 3.12 (`r7-codec-fix.md`). It still needs a decoder pin so lossy JPEG/J2K keep using Pillow (r7 critique), and
    a Kaggle-side check.
- **The only untested lead for the weakest label (ACL) is supervision, not input.**
  - The LM labels' confidence tiers (0.68/0.82/0.94) track severity, and gold-58 counts only complete ACL tears (tier 0.94).
  - Tier-retargeted heads give gold ACL +0.063 [−0.030, +0.169] on the 49-study subset. This is **not confirmed**: the subset
    overlaps the set that suggested the idea, and the tier-restricted LM-test readout is only +0.005.
  - Ticket 05 derives the targets from report text and pre-registers an evaluation on a new set.
- **Measurement lessons added:**
  - Head-seed noise alone is ≈0.002 macro.
  - A 5-fold OOF over the 3,864 non-test studies is a valid second readout for U, but **confounded for A**, because 89% are
    encoder-training studies.
  - Single-readout signs at |Δ| < 0.01 flipped between LM-test and OOF in several arms.

## Executive summary (v5, rounds 1–4)

- **What the model sees now:** 3 far-apart slices per plane packed as RGB, about 31 mm apart, which is about 10% of the slices.
  On the **frozen** DINOv2, un-packing them and adding slices helps a lot: +0.048 LM-test macro with all ~19 band slices.
- **Where that frozen gain comes from:** un-blending (+0.018) and several **distinct neighbouring** slices (+0.022 for 9 vs 1).
  Jittered copies of one slice recover only ~22% of it. Where the slices sit in the band does not matter.
- **On the deployed (contrastively adapted) encoder the gain is small and doubtful.** The best multi-view input (B3, 5 RGB
  groups) adds +0.009 [−0.001, +0.019] against an equally refit baseline, which is below the 0.011 detectable effect, and it
  lands in diffuse labels (effusion/OA). Against the actual deployed bundle it is ≈+0.006 (indicative only). Gold-58 moves the
  other way: −0.025 [−0.049, +0.000], mostly ACL/menisci.
- **Jitter-TTA on the current input** is a free drop-in but gains only +0.002 (not significant) with the current heads.
- **Contrastive adaptation (+0.05 over frozen) is worth more than any input change tested.** No input variant helped the
  weakest labels (ACL, MCL, menisci) on the deployed encoder.
- **3D-native is not the right move** for this data (2D multislice, ~7× anisotropy vs acquired resolution). Also
  deprioritised: higher resolution, crop and intensity changes, coronal T1. Untested: ROI views (H3.1).
- **Not recommended:** a dense-slice retrain, or all-band inference. On the efficiency track B3's extra runtime costs ≈0.06
  AUC-equivalent, if the transcribed scoring formula is right (as written its denominator is negative; re-check it on the
  competition page).
- **Full-train refit gate (pre-registered): FAIL.** With full head-train on the deployed encoder:
  - B3 scores 0.8490 LM-test, +0.0154 [+0.004, +0.026] vs the deployed 0.8336. That clears the +0.011 bar, and the gain is diffuse-led.
  - But gold-58 drops −0.018 [−0.047, +0.008], P(≤0) = 0.90, so B3 fails the gold ≥ 0 bar.
  - The pre-registered consequence: close the multi-view line and spend no submission on it.
  - T9 (TTA) scores +0.006 [+0.001, +0.011] and would also fail.

## Verdict per hypothesis

| ID | Hypothesis | Status | Key evidence | Confidence |
|---|---|---|---|---|
| H1.0 | 3D-native encoder is the wrong move | **Deprioritised** (settled on priors, untested) | 2D multislice, ~3.6 mm slices, ~7× anisotropy vs acquired in-plane; pretrained 3D encoders are CT/brain/isotropic; only ~3.5% of selected series are 3D | Medium |
| H1.1a | RGB channel blending of far-apart slices hurts | **Supported on frozen; absorbed by adaptation** | U: 1 gray slice − RGB triplet +0.018 [−0.000, +0.036]. A: −0.015 [−0.030, −0.002] (A was trained on triplets) | Medium (U); high that it's irrelevant for the deployed encoder |
| H1.1b | Band coverage: lesions missed by 3 sparse slices | **Not supported** | U spread9 − adjacent9 −0.002 [−0.008, +0.004]; all-band − adjacent9 −0.001. The focal-vs-diffuse check was underpowered (CI ±0.03) | Medium-high on U. A peripheral-only focal effect of ~0.01 isn't excluded |
| H1.1b′ | Local slice multiplicity: distinct neighbouring slices beat jittered looks at one | **Supported on U; underpowered on A** | U double difference +0.0175 [+0.006, +0.030] (jitter/adjacent ratio 0.22); A +0.009 [−0.003, +0.022] | Medium |
| H1.2 | Higher input resolution (336/448) | **Not supported inference-only** (r5, r7) | A 336 px +0.0084 [+0.001, +0.016] LM-test (Holm p 0.20), +0.006 vs deployed, gold +0.002: below the gate. U LM-test −0.005, but U OOF +0.007 [+0.003, +0.010]. The A OOF readout is confounded | Medium. Retrain-only candidate (ticket 07) |
| H1.3a | Current 20/50/80 placement is the right anatomy | **Not supported** (placement doesn't matter at a fixed count) | spread ≈ adjacent at k = 3/5/9 | Medium |
| H1.3b | mm-anchored knee-centre slice triplets | **Untested** (blocked: cross-series geometry unverified, FrameOfReferenceUID differs 80/80) | — | — |
| H2.1 | Sequence-aware series selection (reject thin 3D, prefer FS) | **Not a score lever; retrain-time hygiene only** | Changes 45/432 test studies. Swapping with stored heads: −0.022 [−0.062, +0.015]. The subgroup's −0.09 deficit is site/case-mix confounded (Siemens Avanto-fit) | Medium |
| H2.2 | Coronal T1 as a 4th plane (Fracture) | **Deprioritised, untested** | Fracture has 26 LM-test positives (needs Δ ≥ 0.06) and is 0.849 already; a 4th slot crashes on the ~26% of studies without one; metadata-only head 0.59–0.61 | Medium |
| H2.3 | Knee-centred in-plane crop | **Not tested properly** | The r5 Otsu centroid tracked bright signal (cysts, fat-sat edges; 19% of sagittal hit the 20 mm clamp). A +0.007 [+0.001, +0.013], gold −0.006; U +0.005 n.s. No label leak (offset AUROC 0.43–0.58) | Low. It needs a real localiser (07) |
| H2.4 | Intensity normalisation (z-score/N4/Nyúl/CLAHE) | **Not run; predicted null** (closed) | 1–99% window already cancels the 43× scale spread; per-series window ≈ per-slot (+0.001). r5 research found no evidence it matters for pretrained-ViT features | Medium |
| H2.5 | PD-FS vs T2-FS acquisition strata | **Underpowered, descriptive** | −0.030 [−0.076, +0.024]; MDE ≈0.05; case-mix confounded | — |
| H3.1 | Zoomed ROI views (menisci/notch) | **Not supported at inference** (r6) for the image-centred 130+90+65 mm pyramid | U macro +0.014 [+0.002, +0.027] (Holm 0.069), focal +0.002; mechanism vs the off-centre control not supported on focal (+0.006 [−0.011, +0.021]), borderline on U macro (+0.011 [0.000, +0.022]); U OOF +0.0035. A +0.004, ≈ control. Pooled tokens (C-tok) A −0.008 | Medium for inference; a localised ROI is untested |
| H3.2 | Peripheral slices help MCL/Baker's | **Underpowered; not supported on A** | A/B4 MCL −0.017 [−0.052, +0.017], Baker's +0.005 | Low |
| H1.4 | Antialiased resize | **Neutral** (r5) | U −0.001 [−0.004, +0.002], A −0.001 [−0.003, +0.000] | High. Hygiene only, at retrain |
| H1.5 | Fixed zoom (crop 100/115/150 mm) | **Not supported** (r5) | All within ±0.005 macro on U and A, and no focal gain; A/Z100 gold −0.018 | Medium |
| H4.1 | Kaggle decode path silently degrades JPEG-Lossless inputs | **Supported** (r4b, r5, r7) | No JPEG-LL/JPEG-LS decoder on Kaggle; nearest-slice substitution or a crash; ≈ −0.005 per 10% of planes (sim); the vendored-wheel fix passes a bit-exact self-test on 3.12 | High. Tickets 01–03 |
| H4.2 | ACL headroom is partly supervision (LM severity tiers vs gold) | **Open lead, not confirmed** (r5–r7) | 22/23 gold ACL positives are tier 0.94; all 9 LM false positives are low-tier. Tier heads: gold-49 +0.063 [−0.030, +0.169], but the subset overlaps the discovery set; tier-LM-test +0.005 | Low. Ticket 05 |

U = frozen unadapted DINOv2-small (decides mechanism). A = deployed LoRA encoder (decides deployment).

## Next steps (v6): tickets in `.scratch/image-preproc-next/issues/`

| # | Ticket | Status |
|---|---|---|
| 01 | Vendor JPEG-Lossless/JPEG-LS decoders into the Kaggle bundle, with a self-test. Pin Pillow for lossy JPEG/J2K | ready-for-agent |
| 02 | Decode-failure counters, strict mode, and fallback to the next candidate series instead of a crash | ready-for-agent |
| 03 | Private Kaggle run to check the codec bundle (no submission) | ready-for-human |
| 04 | `PREPROCESSING_VERSION` / source-hash cache key, deterministic tie-break, thin-3D rule, multi-frame guard (before any retrain) | ready-for-agent |
| 05 | Report-derived tier-aware ACL/MCL targets, evaluated on a new set that is not gold-58 | ready-for-agent |
| 06 | Whether to spend a leaderboard submission on tier-aware heads (after 05) | ready-for-human |
| 07 | Retrain-only input candidates: gray single slices, multi-scale zoom, 336 px, a localised ROI | needs-triage |
| 08 | Closed lines, not to be re-run | wontfix |

**User decisions (2026-09-27):**
- Ticket 03 is approved: a private Kaggle check, no submission. **Done (r8), 10/10 PASS.**
  - The wheels install offline in 3 s. The self-test and the bundle's own decode are 6/6 bit-exact on the JPEG-LL, J2K and
    JPEG-LS test files, and the example studies are unchanged.
  - But without a Pillow pin, lossy JPEG changes decoder and pixels, so the pin in ticket 01 is mandatory.
  - See `.scratch/input-preproc-hypotheses/r8-kaggle-codec.md`.
  - **Full-bundle check done (r9), 15/15 PASS.** The rebuilt bundle `lm-mlp-tower-size-58-dinov2-small-e5-base-codecs`
    has the same member, with src and codecs from `fix/kaggle-dicom-codecs`. Its CPU CSV is byte-identical to the deployed
    bundle's. On Kaggle (private, no submission) the self-test is 10/10, the lossy decode is unchanged by the pin, the
    example-study hashes are identical, and the decode counters are clean. The fix adds about 5 s per run. It is ready to
    submit at the user's discretion. See `.scratch/input-preproc-hypotheses/r9-kaggle-bundle-check.md`.
- **Main leaderboard only.** The efficiency track is dropped, so runtime is no longer a cost, only the 9 h limit. The runtime
  argument against multi-view (B3) inference is gone, but its gate failure on gold-58 (−0.018) still stands.

Order: 01 → 02 → 03 are robustness work for the current submission. 04 must land before any retrain. 05 is the only open
accuracy lever this study found.

The v5 recommendations below are kept for the record. Items 1, 2 and 6 are superseded by tickets 04 and 01–03 (the Kaggle
decode run happened: r4b).

## Recommendations (v5)

**Can proceed without you** (local only):
1. **Full-train refit gate. DONE: FAIL** (`r4-fulltrain-gate.md`). B3 +0.0154 LM-test, gold-58 −0.018. The text below is the
   original pre-registration. The round-2/3 deltas used a 2,000-study head-train subsample, which overstates the gain over what
   is deployed. Refit the LM-MLP heads on full-train B3 features and on T9 (TTA) features; score against the deployed bundle.
   - **Pre-registered gate:** submit B3 only if B3 − deployed ≥ +0.011 LM-test **and** gold-58 Δ ≥ 0.
   - The expected outcome is that it fails (≈+0.006 expected), which closes the multi-view line.
   - Cost: ~45 min CPU decode plus minutes on the GPU.
2. **Hygiene for the next retrain only; never ship inference-only:**
   - thin-3D reject or slab rule;
   - deterministic series tie-break;
   - antialiased resize;
   - a `PREPROCESSING_VERSION` bump (it was not bumped after the uncommitted crop-or-pad edit, which is a live hazard);
   - a cache keyed on source.
3. **Don't** launch a dense-slice or gray multi-slice contrastive retrain. The A gain is diffuse, at or below the MDE, and gold
   disagrees. **Don't** ship all-band inference (B4): B3 ≈ B4 on A (0.8403 both) with less decode (~5× vs ~6×; 106–147 vs 146–203 min on Kaggle). What would change this: a public-LB gain
   of ≥ ~0.01 from a multi-view submission. If a retrain ever runs, the training form that kept the focal gain on U is random
   *gray single slices* per plane, not RGB K-groups (spread5 − B3 focal +0.023 on U).

**Needs your decision:**
4. **Which track matters: main LB, efficiency, or both.** Efficiency score = AUC/(bench − maxAUC) + runtime/32,400
   (transcribed from the competition page; as written the denominator is negative, so please re-check). 0.01 AUC ≈ 11.7 min of runtime, so B3's extra 69–96 min on Kaggle
   costs ≈0.06–0.08 AUC-equivalent. On the efficiency track only TTA is viable. The main LB ignores runtime.
5. **Whether to override the failed gate and spend a public-LB submission on B3.** The default is no. Case for overriding:
   LM-test (+0.015, CI > 0) is the only local metric that ranked past submissions correctly, and gold-58's CI crosses 0.
   Case against: the gain is in diffuse labels, gold disagrees in sign for the third time on the same features, the
   Kaggle runtime is ~2–3× longer, and the gate was pre-registered. Originally: only if gate 1 passes. Treat |ΔLB| < ~0.008 as inconclusive, and don't
   retrain on an inconclusive result. Confidence that B3 gives a detectable LB gain: low (~25–35%). TTA alone is not worth a
   submission (+0.002), but it can ride along in any future bundle.
6. **Whether to allow a private Kaggle notebook run (about 15 min) to time DICOM decode on the example studies.** The hidden test
   mixes JPEG-2000/JPEG-lossless files, and every local file is uncompressed. This is required before any B3 submission. The current bundle
   evidently runs (0.788 LB), but a failed decode silently substitutes the nearest decoded slice, so the timing run should also
   count decode failures; that is a quiet accuracy risk for the current bundle too.
7. **Where focal-label headroom probably is** (for next planning; a hypothesis, not a result): not input sampling. Candidates
   are the supervision side (LM label noise on ACL/menisci vs gold: LM over-calls MCL, Contusion, Effusion) and ROI views (H3.1).

## Measurement facts that constrain all of the above

- **LM-test has 432 studies.** Paired-bootstrap MDE is **≈0.011 macro**, 0.017 focal composite, 0.024–0.042 per label, and
  0.058 for Fracture (26 positives).
- **Per-label readouts are noisier than macro**, so decisions used pre-registered composites:
  focal = mean(ACL, MM, LM), diffuse = mean(Effusion, MOA, PF OA).
- **LM-test is a thinly validated proxy at the 0.01 scale.** Only one encoder-level pair was ranked correctly, and its CI ∋ 0.
  For the A multi-view arms, LM-test and gold-58 **disagree in sign**.
- **Acquisition metadata alone predicts labels** at LM-test macro 0.59–0.61 (site case mix). Any protocol-sensitive input
  change needs that as a baseline.
- **Heads have constraints:** multi-view features need a head refit (stored heads on B3: −0.018); CPU and cuda heads differ by
  ~0.004, so they are never mixed.
- **Kaggle:** ~1,300 hidden studies, T4, 9 h, offline; external pretrained models allowed. Projected runtime: current input
  37–51 min, B3 106–147 min, B4 146–203 min (all fit; local-decode based, J2K unmeasured). A missing plane raises, so any 4th
  slot must be optional.

## What the current input actually is (round-1 measurements)

| Quantity | Value |
| --- | --- |
| Slices reaching the model | 3 of ~30 per series (≈10%); 20/50/80% of band, ≈31 mm apart |
| Native slice spacing | ≈3.3–3.9 mm; anisotropy ≈7× vs acquired in-plane (≈10–12× vs stored pixel) |
| Model input grid | 130 mm crop → 224 px = 0.58 mm/px; one ViT patch = 8.1 mm |
| In-plane loss vs acquired matrix | median 1.16×, p90 1.44×; 448 exceeds acquired res for 93% of series |
| Resize | bilinear, no antialias; downscale vs stored px median 1.7× |
| Crop centre | image centre; tissue centroid 8–12 mm off (median) |
| Selected sagittal contrast | 78.4% PD-FS, 14.9% T2-FS, 0.07% T1 (vendor-dependent) |
| Series selection | CSV columns + max slice count; thin 3D chosen in 9.4% of studies; CSV-order ties 5.8% |
| Orientation / laterality | consistent; laterality unresolved in 1.7% |

## Experiment log (headline numbers, LM-test macro)

| Round | Experiment | Result |
|---|---|---|
| 2 | Harness validation | Reproduces deployed 0.8336 / gold 0.7659 exactly |
| 2 | Frozen baseline (full head-train) | U 3-slot 0.785 (plane mean) / 0.795 (concat) vs deployed 0.834 |
| 2 | Slice-coverage factorial (7 arms × U/A, head-train 2,000) | U: B1 +0.031, B2 +0.029, B3 +0.040, B4 +0.048 [+0.030, +0.066]; A: B4 +0.010 [+0.000, +0.020] |
| 2 | Critic's view-subset re-scoring (CPU) | Gain from count + un-blending; placement irrelevant; saturates at 5–9 views |
| 2 | Stratified / leak / thin-3D subgroup | PD-FS vs T2-FS n.s.; metadata-only head 0.59–0.61; thin-3D subgroup −0.092 (confounded) |
| 3 | TTA control | A: T9 +0.0045, B3 +0.0088 [−0.001, +0.019] ("mixed"); U: jitter recovers 22% of the adjacent-slice gain |
| 3 | Thin-3D re-selection | −0.022 [−0.062, +0.015] on 45 studies |
| 3 | Critic re-check vs deployed heads | T9 drop-in +0.002; B3 needs refit; ≈+0.006 vs deployed; gold-58 B3 −0.025 |
| 4 | Full-train refit gate (A, n = 3,438) | B3 0.8490, +0.0154 [+0.004, +0.026] vs deployed; gold-58 −0.018 [−0.047, +0.008] → **FAIL**. T9 +0.006, gold −0.003 |
| 4b | Kaggle decode probe (peer session, private kernel) | No JPEG-LL/JPEG-LS decoder; J2K bit-exact via Pillow; per-study decode 0.34 s (order) / 0.42 s (3-slot) / 1.16 s (band) |
| 5 | In-plane factorial, 3 slots (U + A, full-train cuda heads, Holm) | All null. A: R336 +0.008 [+0.001, +0.016], CC +0.007 (gold −0.006), AA −0.001. Head-seed noise 0.0013 (U) / 0.0024 (A) |
| 5 | Code review of decode and parity | Silent JPEG-LL substitution or crash, ≈ −0.005 per 10% of planes; crop-or-pad parity: 5/13,221 series differ (train only) |
| 5 | Research (ROI, resolution, label noise) | Recommended a centre zoom pyramid; found the ACL severity-tier mismatch between LM and gold |
| 6 | Zoom-pyramid ROI views + controls | U Z1 macro +0.014 (Holm 0.069), focal +0.002; mechanism not supported; A +0.004 ≈ C-rand; C-tok A −0.008. Tier side arm: gold ACL +0.109 (partly circular) |
| 7 | Confirmation (OOF cross-fit, tier replication, codec prototype) | A OOF confounded (89% encoder-training studies); U R336 OOF +0.007 [+0.003, +0.010]; tier heads gold-49 +0.063 [−0.030, +0.169], not confirmed; codec self-test 6/6 bit-exact on 3.12 |
| 8 | Kaggle codec wheel check, reduced (private kernel) | Wheels install offline in 3 s; self-test 6/6 bit-exact; example hashes unchanged; lossy JPEG changes decoder without a Pillow pin |
| 9 | Full-bundle Kaggle check (private kernel v2, no submission) | 15/15 PASS: CPU CSV byte-identical to deployed; self-test 10/10; pinned lossy decode unchanged; 9/9 example hashes; decode_counts clean; +5 s per run |
