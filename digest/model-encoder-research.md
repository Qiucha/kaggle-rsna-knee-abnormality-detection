# Encoder research for image–report contrastive learning

Research date: 2026-09-02. This note prioritizes model-owner checkpoints, model cards, and original papers. The model recommendations are for English radiology reports, with MRI/neuroradiology as the closest target domain.

## Recommendation

Use the following first runnable configuration:

- **Image:** DINOv2 ViT-B/14 (`facebookresearch/dinov2`, Hub name `dinov2_vitb14`; Hugging Face equivalent `facebook/dinov2-base`).
- **Text:** `UCSD-VA-health/RadBERT-RoBERTa-4m`.
- **Representation:** DINOv2 class token and the text encoder's first-token representation, each passed through its own trainable projection MLP to a shared 256- or 512-dimensional space and L2-normalized.
- **Optimization:** train the projection heads first with symmetric image-to-text/text-to-image InfoNCE; then, only if validation retrieval plateaus, unfreeze the final one or two blocks of each encoder at a lower learning rate.

This is the most defensible default because the image backbone is a capable, readily available general-purpose feature extractor, while the text model has the closest pretraining domain to MRI reports. The UCSD/VA checkpoint was initialized from RoBERTa and further trained on four million deidentified radiology reports ([official model card](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m)). Its original study used 2.16 or 4.42 million nationwide VA radiology reports and found radiology-adapted variants superior to general, biomedical, and clinical baselines on abnormal-sentence classification, report coding, and report summarization ([original RadBERT paper](https://pubs.rsna.org/doi/10.1148/ryai.210258)).

There is also direct MRI-specific supporting evidence, although it is for report classification rather than image–text retrieval: a 2025 study using 212,588 MRI brain reports found UCSD RadBERT to be the best baseline before local masked-language-model adaptation, and found that adapting every tested encoder on local MRI reports improved its result ([European Radiology article](https://link.springer.com/article/10.1007/s00330-025-11500-9)). Thus, selecting UCSD RadBERT for the first contrastive run is an evidence-based inference, not a claim that it has already been validated for MRI image–report contrastive learning.

## DINOv2 facts that affect implementation

Meta's model card describes DINOv2 as a set of Vision Transformer backbones that return a class token and patch tokens. All released variants use patch size 14; ViT-S/B/L/g have embedding widths 384/768/1024/1536 and approximately 21M/86M/300M/1.1B parameters, respectively. The models were trained on LVD-142M with DINO self-distillation, iBOT masked-image modeling, and KoLeo regularization ([Meta DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)). The 86M-parameter ViT-B/14 is the sensible starting balance between capacity and training cost.

Important pipeline consequences:

- DINOv2 is an **image-only** self-supervised backbone, not a pretrained image–text model. The shared-space projection heads and contrastive objective must learn the cross-modal alignment.
- For image-level retrieval, start with the class token. Meta also reports linear classification using the class token concatenated with the mean of patch tokens, so `CLS + mean(patches)` is a useful controlled ablation if class-token-only retrieval is weak ([Meta model card, Direct Use](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#direct-use)).
- Image height and width should be multiples of 14. The official implementation otherwise crops to the next-smaller multiple; 224×224 yields 256 patch tokens plus the class token ([Meta model card, Model Details](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#model-details)). This should be checked after applying the image preprocessing inherited from the baseline notebook.
- Prefer a frozen-backbone baseline. Meta says the released features work well out of the box and recommends full fine-tuning only as a final step when needed ([Meta model card, Downstream Use](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md#downstream-use)). For MRI, however, staged unfreezing remains an experiment because DINOv2 was not medically pretrained.
- Use the plain ViT-B/14 first. Register variants add four learned tokens and primarily target feature-map artifacts; there is no requirement to adopt them for the initial global-retrieval baseline ([Meta model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)).

## Candidate English medical text encoders

| Priority | Checkpoint | Pretraining match | Why test it | Main caveat |
|---|---|---|---|---|
| 1 (default) | `UCSD-VA-health/RadBERT-RoBERTa-4m` | Four million VA radiology reports, all-specialty radiology | Closest available checkpoint to MRI report language; RoBERTa-base-sized and directly loadable with `AutoModel` | Masked-language-model pretraining does not itself make embeddings contrastive; local reporting style may differ |
| 2 | `StanfordAIMI/RadBERT` | BioBERT initialization followed by 4,056,227 Stanford reports from 608,140 patients, covering more than 1,000 exam types and all body areas | Independent radiology-specific initialization; valuable robustness ablation | Published downstream evidence emphasized COVID chest X-ray/CT classification rather than MRI retrieval |
| 3 | `microsoft/BiomedVLP-CXR-BERT-specialized` | PubMed, MIMIC-III, and MIMIC-CXR, followed by chest-X-ray specialization and multimodal contrastive training | Its class token was explicitly trained in a CLIP-like image–text alignment objective, making it a strong contrastive-initialization control | Strong chest-X-ray domain bias makes it a secondary experiment, not the MRI default |
| 4 | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMed abstracts and PubMed Central full text, trained from scratch | Clean biomedical baseline with an in-domain biomedical vocabulary; useful if radiology checkpoint behavior is brittle | Scientific prose is less similar to terse clinical radiology reports |
| 5 | `emilyalsentzer/Bio_ClinicalBERT` | BioBERT followed by all MIMIC-III notes (about 880M words) | Widely used clinical-note baseline; captures clinical abbreviations and style better than literature-only models | MIMIC notes are ICU-wide, not MRI-specific, and the published pretraining maximum sequence length was 128 |

Supporting primary sources:

- The Stanford RadBERT paper states that its pretraining corpus contained 4,056,227 reports from 608,140 patients and more than 1,000 examination types across all body areas ([Journal of Digital Imaging paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9629758/)); the [owner model card](https://huggingface.co/StanfordAIMI/RadBERT) identifies it as a BioBERT-initialized English radiology model.
- Microsoft's CXR-BERT card states that the specialized model was trained in a CLIP-like multimodal contrastive stage using its `[CLS]` representation, and documents gains on radiology NLI and vision–language tasks. It also explicitly identifies the model as English and chest-X-ray-specific ([Microsoft model card](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized)).
- Microsoft's BiomedBERT (formerly PubMedBERT) was pretrained from scratch on PubMed abstracts and PubMed Central full text and was designed as a broad biomedical encoder ([Microsoft model card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)).
- BioClinicalBERT was initialized from BioBERT and trained on all MIMIC-III notes, about 880M words ([author model card](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)).

## Suggested initial experiment matrix

Hold the image backbone, report preprocessing, split, batch construction, projection dimension, and optimizer constant. Change only the text checkpoint:

1. UCSD RadBERT-RoBERTa-4m — primary baseline.
2. Stanford RadBERT — radiology checkpoint replication/robustness check.
3. CXR-BERT-specialized — tests whether prior multimodal alignment outweighs its chest-domain mismatch.
4. BiomedBERT — broad biomedical control.
5. BioClinicalBERT — broad clinical control.

Choose the winner using bidirectional retrieval on the held-out validation split — which is study-disjoint, and cannot be made patient-disjoint, per the [integrity audit](integrity-coverage-audit.md): image→report and report→image Recall@1/5/10, median rank, and mean rank. Report both directions because a model can retrieve text from images well while producing weaker image retrieval from text.

## Text representation and adaptation details

- Load each checkpoint with `AutoTokenizer` and `AutoModel`; do not use the masked-language-model logits as the report embedding.
- Begin with the first special token (`<s>` for RoBERTa, `[CLS]` for BERT) so every model uses a comparable global representation. Run attention-mask-aware mean pooling as one planned ablation; do not silently vary pooling along with the checkpoint.
- Measure token-length percentiles before fixing the maximum length. If reports exceed the model window, prefer section-aware truncation (retain findings/impression) or chunk-and-pool over arbitrary tail loss. Preserve negation, uncertainty, laterality, measurements, and sequence names; these often carry the discriminative content in MRI reports.
- If extra text-only adaptation is affordable, continue masked-language-model training on **training-split reports only**, then run image–text contrastive training. The MRI brain study above found consistent gains from local MLM adaptation—even from relatively small local corpora—but adaptation must not see validation/test reports or patient duplicates.
- Treat several images or slices paired with one report as grouped positives where the data permits. Otherwise, same-study items inside a batch become false negatives under ordinary InfoNCE.

## Decision rule

Ship the UCSD RadBERT + DINOv2-B/14 configuration as the default unless validation retrieval shows either (a) clear superiority from CXR-BERT despite modality mismatch or (b) unstable tokenization/optimization from the RoBERTa checkpoint. Keep at least BiomedBERT as a non-radiology control so gains can be attributed to radiology specialization rather than merely model size or optimizer noise. Any medical claims or clinical deployment decisions require separate, expert-reviewed validation; these sources support encoder selection, not clinical safety.
