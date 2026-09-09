# Knee CLIP feedback: answers, evidence, and next steps

Prepared 8 September 2026 · Analysis and documentation only

The current evidence supports image–report correspondence, not validated recognition of the 12 target abnormalities. The feedback identifies the right scientific gap, but several implementation questions already have answers and some descriptions need correction. The immediate priority is to settle data provenance and the evaluation contract before selecting more experiments.

This digest answers all eight sections of [the original feedback](knee-clip-feedback.md). It distinguishes current source code, saved experimental evidence, scientific interpretation, and proposed future work. No training, model changes, label changes, checkpoint evaluation, or deployment was performed for this review. Existing results below are historical; temperature summaries were calculated from saved metric logs. The original feedback is preserved.

## 1. Decision map

| Feedback issue | Answer now | What remains unresolved |
| --- | --- | --- |
| 3B versus 9B | Resolved for the current computation: three slices are three channels within each plane; DINOv2 receives `[3B,3,224,224]`. | Real-study slice provenance and whether sparse channel stacking preserves useful pathology. |
| Slice and plane pooling | No independent slice mean exists. Spatial patch tokens are averaged; plane attention varies by study. | Suitability for focal findings and comparison with independent slice encoding. |
| Projection head | Learns features useful for the contrastive target; no direct 12-label supervision. | Whether pathology information improves or is lost at each representation stage. |
| Text truncation and pooling | Padding is masked. Baseline validation truncates 67/441 reports; 43 lose a complete late summary heading under the preflight heuristic. | Clinical content lost, section extraction reliability, and benefit of preserving it. |
| Temperature | Both outputs normalized; shared learned temperature starts at 0.07; inverse scale capped at 100. | Future semantic-target scaling must be specified separately. |
| Semantic supervision | Not implemented in the inspected objective. A report-derived target would be a MedCLIP-inspired adaptation. | Label ontology, missingness, parser validity, normal-case treatment, matrix validity, and independence from evaluation. |
| Translation/model comparisons | Several controlled comparisons and paired intervals already exist. | Translation quality itself is not established; complete matched translation comparisons and text error interpretation are missing. |
| Duplicate/conflicting reports | One confirmed duplicate group with two training entries, both quarantined. | Upstream root cause, exhaustive raw-image/join coverage, and patient independence. |
| 58-study evaluation | A 47/11 split is too unstable for the intended 12-label comparison. | Gold provenance, per-label counts, exposure to representation training, and a defensible exploratory OOF protocol. |

“Resolved” here means the specific implementation question has an evidence-backed answer. It does not mean the design has been shown to be clinically effective.

## 2. Image encoder: the exact tensor flow

**Answer: 3B is intentional for the existing architecture.** The feedback's 9B expectation applies to an architecture that encodes every grayscale slice independently. This project instead uses a three-slice, channel-stacked input for each plane. These are different designs, not equivalent reshapes.

| Stage | Shape | Meaning |
| --- | --- | --- |
| Study batch | `[B,3,3,224,224]` | Study × plane × slice/channel × height × width. Plane order: sagittal, coronal, axial. |
| DINOv2 input | `[3B,3,224,224]` | Row `b*3+p` is one plane of one study. Its three channels contain sampled slices. |
| Patch convolution | `[3B,384,16,16]` | The 14×14 patch projection mixes all three channels. Slice identity no longer has a separate axis. |
| Transformer output | `[3B,257,384]` | One CLS token plus 256 spatial patch tokens. |
| Plane representation | `[3B,768]` | Concatenate CLS with the mean of spatial patch tokens. |
| Restore planes | `[B,3,768]` | Add learned plane embeddings. |
| Plane attention | `[B,3]` | Score each study's plane features and softmax over its three planes. |
| Study representation | `[B,768]` | Weighted sum of that study's plane features. |
| Shared-space projection | `[B,256]` | LayerNorm → Linear → GELU → Linear → L2 normalization. |

There is **no mean over three independently encoded slice features**. Consequently, “replace slice mean with max” is not a small pooling change to the present model. It first requires choosing an independent-slice representation, for example nine grayscale-to-three-channel inputs. That would change encoding and compute as well as aggregation. Hold slice positions and other settings fixed to interpret such a future comparison.

Plane scoring uses shared learned parameters, but its three weights depend on each study's features. The feedback's concern about one global plane weight therefore needs qualification. The actual limitation is that the resulting study embedding is shared across all labels: there is no explicit disease-conditioned plane aggregation. Attention weights alone are not evidence that a lesion was localized.

**Exact selection rule.** For each plane, the source selector prefers fluid-sensitive series and takes the one with most DICOM files; if no fluid-sensitive series exists it falls back to the plane's other series. The first manifest row wins equal file counts. For a geometrically ordered series of length `n`, compute `low=int(0.20*(n−1))`, `high=int(0.80*(n−1))`, then take three evenly spaced positions between these endpoints, cast to integers and deduplicate. If the endpoints do not span a range, use the central slice. Repeat the last index until three slots exist. These are approximately 20/50/80% positions, not three adjacent central slices. Existing cohort records contain selected paths; the selector describes how those paths are constructed, not a new selection performed in this review.

Geometry ordering uses image position projected onto the slice normal where available, then SliceLocation, InstanceNumber, and deterministic UID/file fallbacks. A failed selected-slice decode is replaced with the nearest successfully decoded selected slot; mismatched dimensions become zeros; a fully undecodable plane fails loading. A conditional 130 mm center crop, 1st–99th percentile normalization over the selected stack, and spatial resize follow. Right-knee sagittal processing reverses channel order; coronal/axial processing flips spatial width. DINO input also receives the existing three-channel normalization.

**Evidence boundary.** The saved shape audit used random DINOv2-small weights and two synthetic studies, preserving all 18 input markers. It verifies axis handling, not nine distinct successfully decoded source slices in every real study, spatial lesion coverage, or trained representation quality. Sparse sampling and crop coverage remain plausible limitations, not demonstrated causes of retrieval errors.

**Closure criterion:** an architecture description is now available; a later provenance review should quantify repeated indices, substituted/zero slices, geometry fallbacks, and crop/selection coverage on real studies before interpreting a sampling ablation. Sources: [shape audit](image-shape-audit.md), [model](../src/contrastive_learning/model.py), [preprocessing](../src/contrastive_learning/preprocessing.py), [data selection](../src/contrastive_learning/data.py).

## 3. Projection head: useful information or pair matching?

**Answer: those mechanisms can coexist.** The heads compress image and text features to 256 dimensions and optimize a paired contrastive loss. Features retained may reflect abnormalities, acquisition characteristics, report style, or other correlated information. Neither dimensionality reduction nor improved retrieval demonstrates that pathology was discarded; neither demonstrates that it was preserved.

A future representation comparison needs precise study-level definitions:

| Representation | Proposed definition | Question answered |
| --- | --- | --- |
| Frozen DINO reference | Fixed, declared aggregation of frozen plane features, before learned plane embeddings/attention. | What does the pretrained image encoder already provide? |
| Learned study representation | The 768-dimensional weighted study vector before projection. | What does trained plane aggregation contribute? |
| Projected representation | The normalized 256-dimensional output. | What changes after the projection head? |
| Adapted counterparts | The corresponding stages after image final-block adaptation, and eventually validated semantic training. | Does changing supervision or the encoder improve pathology-related structure? |

A raw collection of three plane vectors is not directly comparable with one study vector until its aggregation is specified. A 768-dimensional probe also has more parameters than a 256-dimensional probe; use the same predeclared regularization approach and report dimensions. Any scaling or dimensionality reduction learned for the probe must use only its training fold.

Use exact-pair retrieval for paired correspondence, gold-label neighbourhood agreement for pathology-related structure, and a strongly regularized image-only probe for label predictability. Define neighbourhood similarity, neighbour count, exclusion of self/duplicates, and rare-label weighting in advance. Raw image and text features do not automatically share aligned coordinates, so raw DINO-versus-raw E5 cosine retrieval is not a meaningful control without a declared alignment method.

If retrieval rises while both pathology-related measures fall consistently, that would support an objective mismatch. One noisy change on 58 studies is insufficient to identify the mechanism. There is currently no completed gold representation comparison establishing this result. Source: [projection and loss implementation](../src/contrastive_learning/model.py), [current data/evaluation boundaries](handoff/).

## 4. Text tower: what 512 tokens and masked mean actually do

**Answer: padding dilution is not the present bug.** The inspected mean-pooling branch uses:

```python
mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
features = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
```

Only mask-valid positions contribute. This includes valid special/prefix tokens; it does not select clinically important words. The reference uses `intfloat/multilingual-e5-base`, mean pooling and `query: `, with a 512-token total input cap. The exact E5 card recommends masked average pooling and task-dependent prefixes, including `query:` for symmetric similarity/feature use. This is a declared baseline convention, not proof it is optimal for MRI–report alignment. Sources: [TextTower](../src/contrastive_learning/model.py), [E5 model card](https://huggingface.co/intfloat/multilingual-e5-base).

The later preflight provides the requested token distribution on the quarantined cohort:

| Baseline text statistic | Training | Validation |
| --- | ---: | ---: |
| Studies | 3,524 | 441 |
| Reports over 512 | 464 (13.17%) | 67 (15.19%) |
| Median tokens | 269 | 271 |
| 90th percentile | 587.7 | 602 |
| 95th percentile | 690.7 | 685 |
| Maximum | 1,241 | 1,032 |
| Last summary heading not fully retained | 302 | 43 |

These are tokenizer-specific counts, not word counts. The earlier audit reported 300 training headings entirely beyond the cutoff; the later preflight reports 302 not fully retained. Their definitions differ, so they should not be merged into one statistic. Heading loss is a lexical proxy: it does not establish exactly which clinical findings are lost or whether equivalent findings appear earlier. Source: [preflight](english-medical-text-preflight.json), [earlier audit](claude-translated-dfd07db-input-audit.json).

The controlled impression-first run scored 9.6372% versus 10.6576% mean R@1 for original full-report order at seed 2026. This rejects promotion under that experiment's rule; it does not establish that impressions are uninformative. The review also found short-report errors and known section-parser limitations. Source: [ordering comparison](claude-translated-dfd07db-comparison-results.md).

**Proposed answer path:** first build a qualitative inventory of actual retained/lost findings in original, translated, and reordered text, including inline headings and findings–impression contradictions. Only then choose a bounded comparison among validated section parsing, head-plus-tail selection, overlapping chunks, or separate section embeddings. Each changes content coverage or aggregation and needs a declared contract. No option is selected or implemented here.

## 5. Temperature: implementation and saved values

**Answer: temperature is already learned and logged indirectly.** Both projection heads L2-normalize their outputs. The model initializes `logit_scale=log(1/0.07)`, then computes `exp(logit_scale) × image_embeddings @ text_embeddings.T`. Symmetric cross-entropy uses this same matrix and its transpose, hence the same temperature in both directions. After optimizer updates the training loop caps log scale at `log(100)`; the effective temperature is at least 0.01. There is no corresponding lower log-scale clamp in this project's training loop, so no finite upper bound on temperature is imposed by that clamp. Sources: [model](../src/contrastive_learning/model.py), [training](../src/contrastive_learning/training.py).

The following values are reciprocals of recorded `inverse_temperature`, not a new checkpoint evaluation:

| Run | Selected epoch | Temperature at selected epoch | Temperature at final epoch 20 |
| --- | ---: | ---: | ---: |
| Frozen multilingual reference, 2026 | 19 | 0.055005 | 0.054990 |
| Frozen reference, 2027 | 18 | 0.055388 | 0.055277 |
| Frozen reference, 2028 | 13 | 0.058540 | 0.055180 |
| Image-block adaptation, 2026 | 18 | 0.050267 | 0.050147 |
| Image-block adaptation, 2027 | 18 | 0.049576 | 0.049444 |
| Image-block adaptation, 2028 | 17 | 0.049977 | 0.049593 |
| English E5, 2026 | 12 | 0.058375 | 0.053894 |
| PubMedBERT Embeddings, 2026 | 19 | 0.052754 | 0.052739 |

Source logs: [reference](../checkpoints/contrastive/claude-translated-dfd07db-q2-v1/00-full-seed2026/metrics.jsonl), [image comparison run directory](../checkpoints/contrastive/claude-translated-dfd07db-image-block-v1/), [text comparison run directory](../checkpoints/contrastive/english-medical-text-v1/). Selected epochs were identified from saved metrics using the documented R@1, validation-loss, then earliest-epoch rule. Historical execution snapshots should remain the authority for historical source behavior.

Smaller temperature sharpens model probabilities and changes training gradients. For fixed embeddings a positive scalar does not change similarity ranking, so changing temperature alone cannot improve R@1 after training. A future temperature used to normalize semantic targets would be a separate parameter with a separate effect. The original MedCLIP defaults are a reference, not an optimum established for this dataset; see [primary-source notes](knee-clip-feedback-source-notes.md).

## 6. Semantic targets: what must be defined before MedCLIP-style training

**Answer: replacing the identity matrix with unvalidated LLM labels is not yet justified.** The current code uses identity targets with symmetric cross-entropy. A semantic alternative would make some unmatched reports partial positives based on medical overlap. This can address false negatives, but can also teach systematic parser mistakes and erase distinctions between cases.

The original MedCLIP uses diagnosis vectors associated with images and UMLS/MetaMap extraction for report sentences. A correction to the feedback: structured image-associated labels are not necessarily independently expert-adjudicated image labels; CheXpert training labels themselves originate in report labeling. Independence must be audited per dataset rather than inferred from the label format. This project would still be an adaptation, because its report-level 12-label ontology and label provenance differ. Sources: [MedCLIP paper](https://aclanthology.org/2022.emnlp-main.256/), [CheXpert primary paper](https://arxiv.org/abs/1901.07031), [source notes](knee-clip-feedback-source-notes.md).

The paper and current public code also differ: the paper defines cosine label similarity, while the authors’ current loss code uses a signed-label dot product before directional softmax. The cosine formula in the feedback should therefore be called the paper-style formulation, not a verified copy of the public implementation. Pin the exact variant before any future reproduction claim. Source: [authors’ loss code](https://raw.githubusercontent.com/RyanWangZf/MedCLIP/main/medclip/losses.py).

### Label contract

For each of the 12 targets, specify definition, anatomy, laterality, negation, uncertainty, historical/postoperative findings, and handling of contradictions between report sections. Preserve the supporting report phrase and provenance. At minimum distinguish positive, explicitly negative, uncertain, and not mentioned. An LLM confidence number is not automatically a calibrated probability. A competition label may use a closed-world convention; a clinically faithful report parser must not silently assume it.

Using a paired report's pseudo-label as an image label is weak supervision. It is not automatically evaluation leakage, but the two labels are not independent corroboration. It becomes leakage when evaluation cases or their gold outcomes influence target design, representation training, or model selection without being accounted for. A report parser's agreement with image gold also mixes parser error with genuine image–report discrepancy; report annotation and image annotation answer different questions.

### Normal and unknown cases

“No positive finding mentioned,” “verified negative for all 12 targets,” and “clinically normal knee MRI” are different statements. Two verified target-negative cases can be related for this task, while differing in conditions outside the 12 labels. An all-zero vector has undefined cosine; adding epsilon avoids numerical failure but does not define the desired semantics. An explicit no-target-finding indicator is only meaningful when all required targets were assessed sufficiently. Unknown cases should not be automatically grouped with negatives.

Possible future candidates are confidence-masked positive overlap, capped inverse-prevalence weighting, and a deliberately limited normal-normal attraction. Estimate prevalence weights only from the appropriate training data. Predefine zero-overlap/no-observed-label behavior. Similarity based on many shared negatives can overwhelm a decisive positive disagreement; positive-only overlap avoids that mechanism but can ignore clinically meaningful explicit negative evidence. No single candidate is established as best.

### The target distribution needs its own audit

Declare the similarity function, masks, row normalization for image-to-text, column normalization for text-to-image, diagonal treatment, batch composition, and any target temperature. Reverse-direction targets must normalize over the reverse candidates; simply transposing an already row-normalized target does not generally preserve row sums. Even symmetric similarities can yield asymmetric normalized probabilities.

A useful mathematical caution: if candidate scores lie in `[0,1]`, one score of 1 against 31 scores of 0 has softmax probability `e/(e+31) ≈ 8.06%`, versus 3.125% uniform probability. Thus “soft medical positives” may be a very diffuse target. This is an illustrative derivation, not a measured project result. Inspect target entropy, diagonal probability, off-diagonal mass, rare-label contradictions, and batch sensitivity. A diagonal/semantic mixture is another design option, but its mixing weight and effect would need validation rather than automatic adoption.

### Validation before training

First assess extraction against independent report adjudication where possible: per-label errors, missingness, uncertainty, and contradiction handling. Then compare a small number of predeclared candidate matrices with a gold-label matrix on eligible expert-labeled studies: pairwise rank agreement, semantic-neighbour precision/recall, and qualitative disagreement review. The gold matrix is a 12-target reference, not exhaustive clinical similarity. Define its metric and treatment of unknown/negative cases too.

Fifty-eight studies give 1,653 unordered pairs, not 1,653 independent observations. Resample studies or known patient groups and rebuild pair statistics; do not independently bootstrap matrix cells. Repeated sampled copies must not be credited as new independent perfect-match neighbours. Use held-out or nested development rules if gold cases influence matrix choice; otherwise call the exercise descriptive development and reserve new independent labels for later validation. Matrix agreement and improved image-only predictions are separate milestones.

**Closure criterion:** a reviewable target specification and a defensible validation design, followed in a separately authorized stage by evidence that the target is faithful enough to test. The present task stops at understanding and planning.

## 7. Translation, encoders, seeds, and retrieval interpretation

**Answer: some recommended controls are already present; the complete translation question remains open.** Comparing original reports with multilingual E5 against translated reports with English E5 changes two factors. It cannot isolate translation. The more recent encoder screen holds translated text fixed and changes the encoder package, including its appropriate tokenizer/prefix. Equal token caps do not mean equal retained text.

| Existing comparison | Result | Supported interpretation |
| --- | --- | --- |
| Full order vs impression-first, seed 2026 | 10.6576% vs 9.6372% mean R@1 | This reordering did not meet its promotion rule. |
| Frozen vs image final block + final norm, seeds 2026/2027/2028 | Gains +0.4535/+1.4739/+1.7007 percentage points | Consistent exploratory direction; no pathology result. |
| Multilingual vs English E5, same translated text, seed 2026 | 10.6576% vs 8.8435%; difference CI −4.4218 to +0.6803 pp | Candidate did not meet its repeat rule; not proof of general inferiority. |
| Multilingual vs PubMedBERT Embeddings, same translated text, seed 2026 | 10.6576% vs 9.4104%; difference CI −3.9683 to +1.3605 pp | Same limitation; this is the specific NeuML embedding model, not every PubMedBERT configuration. |

Across the three image-comparison seeds, frozen-reference mean R@1 is approximately **9.4860% ± 1.0162 pp sample SD**; adapted mean is **10.6954% ± 0.3645 pp**. The paired gain is **1.2094 ± 0.6643 pp**. These summarize three trained seeds, not confidence intervals for deployment performance. All three per-seed paired-query intervals include zero. Repeats were conditional on a positive first-seed result; that screening history limits broad claims. Sources: [image results](claude-translated-dfd07db-image-block-results.md), [text results](english-medical-text-results.md), [ordering results](claude-translated-dfd07db-comparison-results.md).

**Gallery is resolved:** validation concatenates embeddings across batches and compares all 441 studies. Training uses in-batch negatives, normally 31 per query at batch 32. Validation random R@1 is `1/441 ≈ 0.2268%`, not `1/32`. R@5 and R@10 chance levels are about 1.1338% and 2.2676%. Exact ties receive uniform expected credit, which avoids credit determined by arbitrary tied ordering. Identical reports remain separate paired targets; a semantically acceptable but differently paired report can still count as a miss. Sources: [evaluation](../src/contrastive_learning/evaluation.py), [training metrics](../src/contrastive_learning/training.py).

Directional R@1/5/10 and mean/median rank already appear in results. MRR is optional additional evidence, not a substitute for pathology evaluation. Paired query bootstraps preserve method pairing on the same gallery but describe uncertainty conditional on those selected checkpoints and that gallery. They do not capture a newly sampled gallery, unknown repeated patients, seed variation, or all checkpoint/model selection. Combine the two directions within study when estimating their mean, rather than pretending there are 882 independent studies.

**Remaining translation answer:** fix multilingual E5 and the same cohort/splits, compare original, raw translation, and cleaned translation at matched seeds if a future controlled experiment is warranted. First audit translation fidelity on negation, anatomy, laterality, uncertainty, severity, and omissions using source-language review. Retrieval parity alone cannot show that translation quality is solved. The pending qualitative English/medical-text error packet should be interpreted before another candidate is selected; automated worst-case extraction is not that review. Source: [current handoff and pending review](handoff/).

## 8. Identical images with conflicting reports: what is and is not known

**Answer: the known incident is one duplicate pair, containing two entries.** The existing source verification found identical ordered decoded pixels across 30 sagittal, 27 coronal, and 30 axial slices: all 87 slices of the three selected series. Fresh preprocessing matched both caches; report hashes differed. Both DICOM studies were tagged right, while one report stated left. Both entries were training members and are now ineligible in the quarantined cohort. This is not evidence of two separate duplicate groups. Source: [duplicate verification](claude-translated-dfd07db-duplicate-verification.json).

Because decoded source pixels already match, the duplicate cannot be explained solely by the later selected-slice reshape or a stale preprocessed cache. It could still originate in upstream acquisition duplication, exporting, identifier assignment, or image–report joining. Pixel agreement does not identify the correct report or justify choosing one. Quarantine contains the known contradiction without resolving provenance.

Existing coverage is 3,967 train/validation cached tensors plus a 16×16 pooled near-duplicate screen; full decoded-pixel comparison was performed for the known pair. The screen found only that pair, but is not an exhaustive raw-volume audit. Existing corpus audits also report 20 exact translated-report groups spanning splits, involving 78 studies. Repeated text can reflect templates and does not alone prove duplicate patients or images. The locked test's pixels, report examples, labels, and performance were not inspected by these audits. Sources: [input audit](claude-translated-dfd07db-input-audit.json), [provenance continuation](claude-translated-dfd07db-continuation.md).

**Proposed provenance packet:** map source report ID and versions through study/series/SOP identifiers, acquisition metadata, laterality, raw file hashes, ordered decoded-pixel hashes, selected indices, preprocessed hashes, and eligibility/split membership. Separate byte-identical files, identical decoded images, near-duplicate acquisitions, template reports, and many-to-one joins. Record raw and translated report lineage separately. If source patient identity is missing, say so; synthetic grouping IDs do not establish independence.

**Closure criterion:** explain the known pair's lineage or explicitly document why it cannot be recovered; state full audit coverage and exceptions; assess whether historical results need qualification or replacement. Do not silently repair joins, relabel a report, or treat removing two entries as a clean bill of health. Any future whole-dataset integrity check involving the locked test needs a predefined separation from outcome/model-selection access, preferably through an independent custodian. This review does not open it.

## 9. Fifty-eight gold studies: the question the evaluation can answer

**Answer: use all eligible gold studies efficiently for exploratory screening, but do not label the result independent clinical validation.** The 58-case count is user-confirmed in the project record; this review did not re-adjudicate cases or inspect locked labels. The existing retrieval split has only eight gold validation cases and zero positive MCL cases. That is different from the feedback's proposed new 47/11 split; neither supplies an adequate 12-label final test. Source: [data boundaries](claude-translated-dfd07db-continuation.md).

A 256-to-12 linear layer has 3,072 weights plus 12 intercepts. Strong regularization reduces effective complexity but cannot replace missing independent examples. High prediction confidence is not representation quality or reliable calibration.

### Resolve independence before choosing folds

Record each gold study's image-label provenance, adjudication, missing labels, duplicate/patient group where known, and exposure to contrastive training, report-based targets, validation checkpoint selection, and previous manual analysis. Holding a case out of a probe does not undo its earlier exposure to the representation learner.

Two estimands must be distinguished: a frozen-representation probe on this available cohort, potentially exposed during pretraining, and generalization to new independent patients. The former may be feasible as exploratory analysis. The latter requires independent data or a properly isolated representation-development process, not just a new probe split. Using all 58 must also respect any current locked-test membership; that conflict needs an explicit protocol decision before data access.

### Proposed probe contract

| Element | Proposed decision and limitation |
| --- | --- |
| Model | Per-label, strongly L2-regularized logistic probes with predeclared settings; no MLP or threshold search. |
| Features | Small prespecified set of frozen study representations; same folds for all methods. |
| Preprocessing | Fit standardization, imputation and any dimensionality reduction inside each training fold. |
| Folds | Prefer feasible grouped multilabel folds after inspecting eligible label counts; LOO is an alternative, not automatically best. Keep known duplicate/patient groups together. |
| Rare labels | Check both test and training class counts. A label with one positive can leave a training fold with no positives. Mark it not estimable under the probe protocol; do not fabricate a score. |
| Predictions | In one fold partition, exactly one held-out prediction per study. With repetitions, retain a study-by-repeat table; never count repeated predictions as additional studies. |
| Metrics | Report positive/negative/unknown counts, per-label AUROC where support exists, and explicitly named macro-average eligibility. Consider PR metrics with prevalence context if predeclared. |
| Uncertainty | Resample known patients, otherwise studies with an explicit independence limitation; pair methods on the same resampled units. Report invalid rare-class bootstrap draws and coverage. |
| Tuning | Fix hyperparameters beforehand or tune entirely within outer training folds. Keep gold-driven target design and model selection inside the same independence boundary. |

Pooled OOF AUROC is preferable to calculating AUROC on a single held-out LOO case, which is undefined. However, pooled scores come from different fitted models. AUROC compares scores across cases and is not a per-case decomposable metric; changes in model scale or fold prevalence can distort the pooled ranking. Report the construction and comparability assumption and treat it as an exploratory statistic, not an automatically unbiased generalization estimate. Where fold-level AUROCs are defined, they can be a declared sensitivity analysis. See [scikit-learn's cross_val_predict caveat](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) and [source notes](knee-clip-feedback-source-notes.md).

Bootstrapping fixed OOF predictions describes conditional case-level variability. It does not repeat representation learning, all probe training, matrix selection, or architecture search. Refitting the complete procedure would answer a broader uncertainty question but is not automatically reliable with only 58 cases either. Cross-validation and bootstrap redistribute or resample existing information; neither creates new patients.

**Closure criterion:** a signed-off exploratory estimand, eligible-case inventory, exposure audit, per-label feasibility table, folds and selection rules before fitting anything. If the same 58 cases are used to choose a semantic target and report its success without nesting or separate labels, call that development evidence. Additional independently adjudicated studies are needed for a credible final estimate.

## 10. Proposed next steps and decision gates

This sequence is a planning recommendation. No job is queued and no implementation is requested by this document. It replaces “try another model next” as the proposed priority for responding to this feedback; it does not change existing frozen experiment protocols.

| Order | Question to settle | Concrete reviewable output | Exit condition |
| --- | --- | --- | --- |
| 1 — Integrity | Are the study/report pairings and real image inputs trustworthy enough to interpret results? | Provenance packet for the known pair; join and duplicate coverage table; selected-slice failure/fallback inventory. | Every claim distinguishes verified coverage from unknowns; systematic contradictions are resolved or explicitly constrain interpretation. |
| 2 — Evaluation contract | What independent evidence can the gold cases provide? | Gold-case provenance/exposure inventory, per-label feasibility counts, intended estimand, fold/uncertainty plan. | No implicit patient-independence, weak-label ground truth, locked-test reuse, or hidden gold-driven tuning. |
| 3 — Explain existing results | Which remaining errors suggest lost content, report inconsistency, or objective mismatch? | Qualitative text-candidate review and retained/lost finding inventory, compared with completed ordering/image reviews. | Hypotheses are tied to actual examples and separated from causal claims; no need to rerun completed comparisons merely to reproduce tables. |
| 4 — Specify semantics | What should two studies sharing a finding mean to the objective? | Twelve-label parsing/uncertainty contract, normal-case policy, small candidate-matrix specification and validation protocol. | The target is mathematically defined for all cases and its gold-data use is compatible with step 2. |
| 5 — Choose a minimal future comparison | Which single unresolved hypothesis is most useful to test? | A bounded protocol specifying comparator, one intended change, fixed seeds/gallery/checkpoint rule, metrics and stopping condition. | Integrity and evaluation limitations are accepted explicitly; matrix validation precedes semantic training. No broad architecture sweep. |
| 6 — Strengthen final evidence | How will results eventually generalize beyond these cases? | Plan for additional independent expert labels and a held-out final evaluation. | A final assessment is separated from representation, target, and threshold development. |

Steps 2 and 3 can be developed while provenance questions are investigated. Defining the evaluation contract belongs before semantic-matrix tuning, even if actual downstream fitting happens later. Frozen DINO versus current CLIP representation screening need not wait for a MedCLIP implementation; it does require the exposure and label contract. Pooling, chunking, more slices, disease-specific attention, and text adaptation remain candidate hypotheses rather than a mandatory implementation checklist.

The near-term deliverable is therefore an evidence and evaluation decision, not a more complex model. Retain the multilingual full-report model as an experimental reference, describe image adaptation as promising, and leave final model selection open.

## 11. Evidence guide and remaining unanswered questions

| Evidence | Use it for |
| --- | --- |
| [Original feedback](knee-clip-feedback.md) | The eight concerns and original recommendations. |
| [Current project digest](project-state.html) | Broader architecture and experiment history. |
| [Shape audit](image-shape-audit.md) | Synthetic runtime evidence and real-data limitations. |
| [Model](../src/contrastive_learning/model.py), [preprocessing](../src/contrastive_learning/preprocessing.py), [training](../src/contrastive_learning/training.py), [evaluation](../src/contrastive_learning/evaluation.py) | Current computation, pooling, normalization, temperature, gallery and objective. |
| [Input audit](claude-translated-dfd07db-input-audit.json), [duplicate verification](claude-translated-dfd07db-duplicate-verification.json), [continuation](claude-translated-dfd07db-continuation.md) | Scope of integrity checks and quarantine. |
| [Text preflight](english-medical-text-preflight.json) | Token distributions and model-specific input coverage. |
| [Ordering comparison](claude-translated-dfd07db-comparison-results.md), [image results](claude-translated-dfd07db-image-block-results.md), [text results](english-medical-text-results.md) | Completed comparisons, directional metrics, uncertainty. |
| [Current handoff](handoff/) (local) | Pending qualitative text review and historical execution boundaries; experiment suggestions there do not authorize work in this session. |
| [Primary-source notes](knee-clip-feedback-source-notes.md) | MedCLIP/E5 provenance and OOF methodology qualifications. |

The repository cannot currently answer which report belongs to the known duplicate acquisition, whether all remaining studies are independent, how much target-specific clinical information truncation removes, whether the projected embedding preserves all 12 targets, which semantic target is reliable, or how accurately a final image-only model generalizes. Each is retained above as an explicit question with an evidence requirement, rather than filled in with an assumption.
