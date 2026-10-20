# Knee CLIP feedback — primary-source checks

Research date: 2026-09-08. These notes support the feedback digest; they do not authorize or implement experiments. Local pipeline behavior must be established from local code and artifacts separately.

## MedCLIP: distinguish the paper, public code, and proposed adaptation

The paper aligns image diagnosis labels and report-sentence entities in a shared 14-concept vocabulary using MetaMap/UMLS. Equations 3–5 specify cosine label similarity followed by separate row softmaxes for each retrieval direction. Equation 7 applies temperature to the predicted embedding similarities; the target softmax has no explicit temperature. [MedCLIP paper, §§3.2–3.3](https://aclanthology.org/2022.emnlp-main.256.pdf).

The current public loss instead computes a label **dot product**, then separately softmaxes it and its transpose. Its comments describe −1 as negative, 0 as uncertain, and +1 as positive. A docstring mentions clamping, but the shown executable loss does not perform that clamp or cosine normalization. This file also supports ordinary hard CLIP when labels are absent. Consequently, cite the intended variant explicitly and pin a revision before claiming reproduction. [Authors' loss source](https://raw.githubusercontent.com/RyanWangZf/MedCLIP/main/medclip/losses.py).

Structured image labels are not synonymous with independently adjudicated image truth: CheXpert training labels were automatically extracted from reports. A new LLM labeler therefore differs in ontology, extraction rules, uncertainty handling, and validation, but report-derived supervision itself is not unique to this proposed adaptation. [CheXpert paper](https://arxiv.org/html/1901.07031).

**Inference for this project:** assigning the paired report's pseudo-label to the image is legitimate weak supervision, but it creates no independent image-side confirmation. If parser errors affect both sides, apparent semantic agreement can be self-consistent and wrong. Evaluate against expert labels and inspect extraction errors; do not describe this dependence alone as proof the method cannot work.

## Two temperatures and the soft-target dynamic range

The official model learns `log(1 / tau)`, initializes `tau = 0.07`, normalizes image/text embeddings, and clamps log scale to `[0, 4.6052]`, corresponding approximately to `tau` in `[0.01, 1]`. The same logit matrix is transposed for the other direction. These are source facts, not a recommendation for the knee dataset. [Authors' model source](https://raw.githubusercontent.com/RyanWangZf/MedCLIP/main/medclip/modeling_medclip.py).

**Mathematical deduction from the feedback formula:** for nonnegative, nonzero binary labels, cosine similarity lies in `[0,1]`. With `y = softmax(s)`, the highest possible target probability ratio is only `exp(1) ≈ 2.72`. In a batch of 32 with one score of 1 and 31 scores of 0, the best target receives `e/(e+31) ≈ 8.06%`; every unrelated candidate still receives about `2.97%`. Thus this particular target can be surprisingly close to uniform. This bound does not apply to the unnormalized signed-label dot product in the public loss.

A separate target temperature `tau_target` would change that ratio to `exp(1/tau_target)`; it is a new design choice, not the model temperature. Changing model temperature changes predicted probabilities and optimization, while changing target temperature changes desired supervision. Assess target entropy, diagonal/off-diagonal mass, candidate composition, and zero-vector policy before selecting a formulation. Lower model temperature scales similarity gradients by `1/tau`, but does not guarantee every example has a larger gradient because the probabilities also change.

## E5: confirmed defaults and what they do not establish

The `multilingual-e5-base` model card implements attention-mask-aware mean pooling, truncation at 512 tokens, and L2 normalization. Its guidance uses `query:` for symmetric similarity and feature extraction, and `query:`/`passage:` for asymmetric query/document retrieval. The English prefixes apply even to non-English input. [Official model card](https://huggingface.co/intfloat/multilingual-e5-base).

**Inference:** MRI-image/report alignment is a new cross-modal use, so these instructions do not experimentally settle the best report prefix for this pipeline. Preserve the exact checkpoint's convention as a documented baseline and keep it fixed in comparisons. The documented 512-token setting says nothing about how often this dataset loses decisive findings; only its tokenized reports can answer that. Translation and prefixes consume token budget too.

## The 58-study evaluation: necessary corrections

Rare classes can produce undefined ROC AUC in validation folds and fitting failures in training folds. Stratification addresses fold construction, not inadequate statistical information, and can understate variation across folds. Repeated studies from a patient must stay together when estimating performance on unseen patients. [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

**Implications for the proposed protocol:**

- Leave-one-out fold AUROC is always undefined because a fold has only one study. Pooled out-of-fold AUROC can be computed if both classes exist overall, but that alone does not make the estimate reliable.
- If a label has one positive patient, the fold holding out that patient has no positive training patient. Do not silently manufacture useful discrimination using a default constant predictor. Document the label as not evaluable for the intended learned-probe comparison, or explicitly report a prespecified fallback and its limitations.
- Repeated folds create several predictions for each study. Specify either one pooled AUROC per repetition or a study-level average of held-out predictions. Never count repeated predictions as additional independent patients.

Pooled cross-validated metrics need not estimate generalization performance in the same way as fold metrics; the API explicitly cautions about metrics that do not decompose over samples. AUROC compares pairs, including scores from different fitted models when predictions are pooled. [scikit-learn `cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html).

**Inference:** different fold intercepts and score scales can distort pooled rankings. Fixed preprocessing/probe choices, identical folds across representations, and an explicitly exploratory interpretation are essential. Patient bootstrap of fixed out-of-fold predictions describes resampling uncertainty conditional on those fits; it does not fully capture refitting, model selection, or overlapping-training-fold dependence. Report undefined bootstrap replicates and the resampling unit rather than implying all confidence intervals are fully validated.

## Protect the gold set before semantic-matrix design

Preprocessing and feature selection learned from evaluation data cause leakage; transformations must be fitted on training data. [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). Selecting settings and estimating their performance on the same data induces selection bias; nested evaluation separates these roles. [scikit-learn nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

**Application to this project:** if all 58 labels select the parser prompt, label ontology mapping, similarity rule, normal-case policy, class weights, or representation/checkpoint, later cross-validating only the final linear probe does not undo that reuse. Decide first whether these cases are development material or protected evaluation material. Either keep all choices fixed independently, nest every data-dependent choice inside outer training folds, or clearly label the resulting numbers development estimates and obtain an independent final set. Tiny samples can make full nesting impractical; that is a limitation to acknowledge, not permission to claim unbiased validation.

Also establish whether held-out patients' images or reports entered contrastive pretraining. A probe can be out-of-fold while its frozen encoder already saw their reports. Such an experiment answers a different, transductive question from unseen-patient generalization. A matrix validated against the same report parser on both axes is parser consistency, not independent clinical validation.
