# Critical Feedback on the Current Knee MRI–Report Contrastive Pipeline

## Executive summary

The current pipeline is a useful and necessary baseline: it shows that non-random image–report retrieval is possible, and the consistent improvement from unfreezing the last DINOv2 block is promising. However, the present evidence supports only the claim that the model learns **image–report correspondence**. It does not yet show that the image representation captures the 12 target abnormalities.

The highest-priority issues are:

1. verify the image tensor flow (`3B` versus `9B`) and the slice/plane aggregation;
2. audit the two identical-image/conflicting-report cases before further modelling;
3. define and validate the MedCLIP-style semantic target independently of downstream training;
4. make the translation/model comparisons controlled and report uncertainty;
5. replace the proposed 47/11 classification split with a strictly exploratory out-of-fold evaluation on all 58 gold-labelled studies.

---

## 1. Image encoder: `3B` versus `9B` and aggregation across slices

### What appears to be implemented

The diagram gives the image input as

$$
X \in \mathbb{R}^{B \times 3 \times 3 \times 224 \times 224},
$$

corresponding to three planes and three slices per plane. If every 2D slice is independently passed through DINOv2, the encoder should initially receive $9B$ images, not $3B$.

### Assessment

**This concern is valid and should be treated as an implementation check, not merely a notation issue.** Please log the tensor shape before and after every `reshape`, `permute`, flattening, and aggregation operation. We need to establish whether:

- all nine selected slices are encoded;
- three slices within each plane are combined before DINOv2;
- any slice or plane is accidentally discarded;
- the batch, plane, and slice axes remain distinguishable after flattening.

### Is mean pooling across slices reasonable?

Mean pooling is a defensible baseline, but it is not automatically appropriate for focal pathology. If an abnormality is visible in only one of three slices, simple averaging can dilute its signal:

$$
v_p = \frac{1}{S}\sum_{s=1}^{S}v_{p,s}.
$$

This should be tested rather than assumed. A minimal controlled ablation would compare:

- mean pooling;
- max pooling;
- learnable attention pooling;
- concatenation followed by a small projection;
- different numbers and positions of selected slices.

The same issue applies to the weighted sum across planes. A single global plane weight may not suit all 12 abnormalities, because different findings may be most visible in different planes. Disease-specific aggregation could be considered later, but only after the current tensor flow is verified.

**Required clarification:** Please document the exact slice-selection rule, the slice-level tensor shape entering DINOv2, the within-plane aggregation, and the across-plane aggregation.

---

## 2. Projection MLP: information extraction versus pair matching

### What the projection head is doing

The image and text features are projected into a shared 256-dimensional space:

$$
z_i^{I}=f_I(h_i^{I}), \qquad z_j^{T}=f_T(h_j^{T}).
$$

The projection heads are not directly supervised to preserve the 12 abnormalities. They are supervised through the contrastive objective, which rewards agreement with the chosen target matrix.

### Assessment

**The concern is valid, but “extracting information” and “forcing a match” are not mutually exclusive mechanisms.** A projection head necessarily discards some information while retaining features useful for its training objective. Under one-hot CLIP supervision, those useful features may include true pathology, but may also include report template, acquisition protocol, laterality, artefacts, or other patient/study-specific cues.

Therefore, the question must be converted into testable comparisons. We should evaluate representations at multiple stages:

1. raw/frozen DINOv2 representation;
2. study-level representation after slice/plane aggregation;
3. the 256-dimensional projected representation;
4. the same representations after naive CLIP and after semantic-target training.

For each stage, compare:

- exact-pair retrieval;
- pathology-aware neighbourhood quality on the 58 gold studies;
- a heavily regularized out-of-fold linear probe for the 12 labels.

If Recall@1 improves while pathology-aware neighbourhood agreement or out-of-fold AUROC worsens, the projection head is optimizing pair identity at the expense of the downstream task. Recall@1 alone cannot resolve this question.

---

## 3. Text tower: 512-token truncation and mean pooling

### Token limit

The 512-token limit is likely inherited from the text encoder rather than justified by the structure of these reports. It is acceptable as a baseline only after reporting:

- the proportion of reports longer than 512 tokens;
- the distribution of token lengths;
- which report sections are truncated;
- whether clinically relevant findings occur after the cutoff.

Moving `Impression` to the beginning with a regex reduced Recall@1 in the preliminary experiment. That result shows only that this particular preprocessing rule did not improve exact-pair retrieval. It does **not** establish that `Impression` is uninformative or that truncation is harmless.

Recommended alternatives include:

- reliable section parsing for `Findings` and `Impression`;
- head-plus-tail truncation;
- overlapping chunks followed by chunk-level pooling;
- separate section embeddings followed by learned aggregation.

These methods should first be compared with the original report order under identical settings.

### Mean pooling over text tokens

Mean pooling is standard for E5-style sentence embeddings and is a reasonable baseline, provided that padding tokens are excluded:

$$
t_i=
\frac{
\sum_{k=1}^{T_i}m_{ik}h_{ik}
}{
\sum_{k=1}^{T_i}m_{ik}
},
$$

where $m_{ik}$ is the attention mask. A naive mean over all 512 positions would incorrectly dilute shorter reports with padding.

Mean pooling can still dilute a short but decisive phrase inside a long report. It should therefore be compared with section/chunk pooling. The `query:` or `passage:` prefix should also follow the exact E5 model's recommended usage and remain fixed across comparisons.

**Required clarification:** Please show the pooling code and confirm attention-mask-aware pooling.

---

## 4. Temperature in cosine-similarity logits

### Why temperature matters

After L2 normalization, the logit is normally

$$
\ell_{ij}=
\frac{
(z_i^{I})^{\top}z_j^{T}
}{
\tau
}.
$$

The temperature $\tau$ controls the sharpness of the predicted distribution:

$$
p_{ij}^{I\rightarrow T}
=
\frac{\exp(\ell_{ij})}
{\sum_k \exp(\ell_{ik})}.
$$

Small $\tau$ produces sharper probabilities and stronger gradients on hard negatives; a larger $\tau$ produces softer discrimination. It therefore interacts directly with false negatives and with the proposed semantic soft targets.

### Assessment

**This is a valid missing implementation detail.** The team should report:

- whether both image and text embeddings are L2-normalized;
- whether $\tau$ is fixed or learnable;
- its initialization and any clamping range;
- its final value;
- whether the same temperature is used in both directions.

The original MedCLIP implementation used a learnable temperature initialized at $0.07$, but that value is a reference point rather than a guaranteed optimum for this MRI dataset. Temperature should not be tuned extensively against the 58 gold cases.

---

## 5. MedCLIP-style correction and construction of the semantic target

### What MedCLIP changes

Naive CLIP treats only the matched study/report as positive:

$$
Y_{ij}=\mathbb{1}[i=j].
$$

MedCLIP instead constructs a medically informed similarity $s_{ij}$ and normalizes it into a soft target distribution:

$$
s_{ij}=
\frac{
l_i^{I\top}l_j^{T}
}{
\lVert l_i^I\rVert_2\lVert l_j^T\rVert_2
},
$$

$$
y_{ij}^{I\rightarrow T}
=
\frac{\exp(s_{ij})}
{\sum_k\exp(s_{ik})}.
$$

The training loss matches the model distribution $p_{ij}$ to the semantic target $y_{ij}$ rather than to a one-hot identity matrix.

### Assessment of an LLM-derived 12-label target

**The concern is correct: directly converting each report into 12 binary LLM labels and then computing similarity is not sufficiently validated.** It introduces at least four problems:

1. **Parser error:** false positives and false negatives become training targets for the image encoder.
2. **Missingness versus absence:** an unmentioned finding is not always a confirmed negative.
3. **Loss of report detail:** the 12 competition labels may omit severity, location, chronicity, surgery, and other clinically relevant distinctions.
4. **Circular supervision:** if the paired report supplies the pseudo-label assigned to an image, the semantic matrix inherits the report parser's systematic errors.

This differs from the original MedCLIP setup, which used structured image labels and UMLS/MetaMap entities extracted from reports. In this project, most images do not have independent structured labels. Therefore, a report-derived 12-label matrix is an adaptation of MedCLIP, not a direct replication.

### The normal-case issue

The statement “normal studies should not be medically similar” is **too strong**. Two studies negative for all 12 target abnormalities are similar with respect to the Kaggle target, so treating them as partially related may help downstream classification. However, they are not necessarily interchangeable as complete clinical studies.

There is also a mathematical failure: an all-zero 12-label vector makes cosine similarity undefined. If all-zero Jaccard similarity is manually set to one, all target-negative studies may collapse into a single strong positive group.

Recommended handling:

- add an explicit and independently verified `no target finding` indicator rather than relying on an all-zero vector;
- distinguish `negative`, `uncertain`, and `not mentioned`;
- avoid allowing negative-negative agreement across many labels to dominate similarity;
- consider positive-finding overlap with inverse-prevalence weighting;
- compare alternatives both with and without normal-normal soft attraction.

A possible class-balanced positive-overlap score is

$$
s_{ij}^{+}=
\frac{
\sum_{c=1}^{12}w_c\min(q_{ic},q_{jc})
}{
\sum_{c=1}^{12}w_c\max(q_{ic},q_{jc})+\varepsilon
},
$$

where $q_{ic}\in[0,1]$ is a probabilistic label estimate and $w_c$ downweights common findings. This is only a candidate and must be validated; it should not be adopted because the formula looks plausible.

### How the matrix should be validated

Before training with it, compare the report-derived matrix with a gold matrix built from the 58 expert-labelled studies:

- rank correlation between pairwise similarities;
- precision/recall of gold semantic neighbours;
- per-label error and uncertainty analysis;
- study-level bootstrap, not pair-level bootstrap, because the $58\times57/2$ pairs are not independent.

Then separately ask whether training with this matrix improves image-only out-of-fold performance. Matrix fidelity and downstream usefulness are related but distinct questions.

---

## 6. Translation experiments and random seeds

### Assessment

**This concern is valid.** Several preliminary comparisons change both the report representation and the text encoder. For example, `original report + multilingual E5` versus `translated report + English E5` does not isolate translation quality.

The controlled comparisons should be:

1. fix multilingual E5 and compare original, raw translation, and cleaned translation;
2. fix the exact English text and compare multilingual E5, English E5, and PubMedBERT;
3. fix preprocessing, split, optimizer, epochs, checkpoint selection, and retrieval gallery;
4. repeat across the same seeds.

The current translation differences are smaller than the observed variation across seeds. Therefore, the claim that translation quality is “mostly solved” is not supported by the current results.

Three training seeds are acceptable for an exploratory signal, especially because unfreezing improved performance in all three, but they are not sufficient for a strong conclusion. Training-seed variation and validation-sample uncertainty are different quantities. Report both:

- mean and standard deviation across training seeds;
- a paired bootstrap over the same validation queries for method differences.

The retrieval evaluation must also state whether the gallery includes all 441 validation studies or only the current batch. With a global gallery the random Recall@1 baseline is approximately

$$
\frac{1}{441}\approx0.227\%,
$$

whereas within a batch of 32 it is

$$
\frac{1}{32}=3.125\%.
$$

Those settings lead to very different interpretations.

---

## 7. Identical images with conflicting reports

### Assessment

**This is a P0 data-integrity issue and must be audited before further modelling.** Simply removing the two observed pairs is insufficient.

Possible causes include:

- incorrect image–report join keys;
- duplicated or amended studies;
- cache/file-name collisions;
- a preprocessing bug that maps different studies to the same selected slices;
- duplicate studies split across training and validation;
- genuinely duplicated acquisitions paired with different reports.

For every duplicate group, inspect:

- patient, study, series, and SOP instance identifiers;
- acquisition date/time and laterality;
- hashes of raw pixel data and preprocessed tensors;
- selected slice indices;
- raw report, translation, and report identifier;
- train/validation membership.

The audit should also scan the full dataset for duplicate raw images, duplicate preprocessed tensors, duplicate reports, and many-to-one joins. If this is a systematic join or caching error, it can invalidate all subsequent results.

---

## 8. Evaluation with only 58 gold-labelled studies

### Assessment

**The proposed 80:20 split is not scientifically usable for a 12-label evaluation.** It leaves approximately 47 training and 11 test studies. Some labels may have no positive or no negative test examples, making AUROC undefined. Even when defined, one case can change the result substantially.

A 256-to-12 linear head already contains thousands of parameters before considering intercepts. Training it on 47 studies without extremely strong regularization is likely to overfit. Prediction confidence is not a representation-quality metric and may be badly miscalibrated.

### What cross-validation and bootstrap can and cannot do

Cross-validation and bootstrap **do not solve the sample-size limitation**. They cannot create additional independent clinical information. They can only:

- avoid wasting most of the 58 cases in one tiny holdout;
- produce out-of-fold predictions;
- quantify how unstable the estimate is;
- support cautious exploratory comparison between a small number of pre-specified models.

The best available evaluation is therefore:

1. freeze the image embeddings;
2. use a strongly regularized linear/logistic probe, not an MLP;
3. generate one out-of-fold prediction per study using leave-one-out or carefully constructed repeated multilabel-stratified folds;
4. aggregate all 58 out-of-fold predictions before calculating each AUROC;
5. report the positive and negative count for every label;
6. use a patient-level bootstrap for confidence intervals;
7. compare all representations using identical folds and fixed hyperparameters.

Even this remains exploratory. Wide intervals, undefined rare-label metrics, and unstable model rankings should be expected. The 58 cases cannot provide a precise final estimate of generalization, nor should they be repeatedly reused for architecture selection, semantic-target design, threshold selection, and final reporting.

Obtaining additional independent expert labels is the only direct solution to this bottleneck.

---

## Recommended order of work

### P0 — verify pipeline integrity

- [ ] Resolve the `3B` versus `9B` tensor-flow ambiguity.
- [ ] Audit the two identical-image/conflicting-report cases and scan all duplicates/joins.
- [ ] Confirm global versus batchwise retrieval evaluation.
- [ ] Confirm masked text pooling, L2 normalization, and temperature implementation.

### P1 — establish controlled baselines

- [ ] Run controlled translation and text-encoder comparisons with matched seeds.
- [ ] Report image-to-text and text-to-image Recall@1/5/10 separately.
- [ ] Add median rank or MRR and paired uncertainty estimates.
- [ ] Compare DINO, aggregated, and projected representations.
- [ ] Ablate slice pooling and selected-slice count.

### P2 — validate semantic supervision

- [ ] Define uncertainty-aware report labels rather than forcing every item to binary.
- [ ] Compare candidate semantic matrices against the 58 gold studies.
- [ ] Decide explicitly how all-negative/normal studies are handled.
- [ ] Train one-hot CLIP and semantic-target variants under identical settings.

### P3 — limited downstream evaluation

- [ ] Produce out-of-fold linear-probe predictions on all 58 studies.
- [ ] Report per-label counts, AUROC where defined, and wide confidence intervals.
- [ ] Treat the result as model screening rather than definitive validation.
- [ ] Seek additional independent expert-labelled studies.

---

## Bottom line

The current work successfully establishes a non-trivial image–report retrieval baseline. The consistent gain from unfreezing the final DINOv2 block is worth following up. However, the project should not yet claim that the learned image embedding is medically aligned with the 12 target abnormalities. The next milestone is not a more complex model; it is a verified data pipeline, a validated semantic target, and an evaluation design that clearly separates exact-pair retrieval from downstream pathology recognition.

## References

- Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Wang et al. (2022), [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://arxiv.org/abs/2210.10163).
- Wang et al. (2024), [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533).

