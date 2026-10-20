# Frozen English and medical sentence encoder shortlist

Research date: 2026-09-07. Scope: two replacements for the current frozen text backbone, keeping the existing image tower and learned 256-dimensional projection. This updates the selection rationale in [the earlier encoder survey](model-encoder-research.md): domain-adapted masked-language models remain useful controls, but the next small screen should compare models explicitly trained for sentence embeddings. No model weights were downloaded and no training or GPU measurements were performed for this research.

## Selected checkpoints and reproducible identities

| Role | Exact repository | Pinned revision, resolved from the public Hub API |
| --- | --- | --- |
| General English sentence encoder | `intfloat/e5-base-v2` | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| Medical sentence encoder | `NeuML/pubmedbert-base-embeddings` | `b79526d6ef3645e0df4530322e266f24c829f5ef` |

Both repositories were public and ungated at inspection. E5 declares MIT; NeuML declares Apache-2.0. Pin tokenizer and model to the same SHA. These are the `main` revisions observed on the research date, not guarantees about future `main`. Source: [E5 pinned Hub metadata](https://huggingface.co/api/models/intfloat/e5-base-v2/revision/f52bf8ec8c7124536f0efb74aca902b2995e5bcd?blobs=true), [NeuML pinned Hub metadata](https://huggingface.co/api/models/NeuML/pubmedbert-base-embeddings/revision/b79526d6ef3645e0df4530322e266f24c829f5ef?blobs=true).

Metadata, model cards, pooling/module configurations, and tokenizer assets are saved locally under `checkpoints/contrastive/english-medical-text-v1-models/{e5-base-v2,pubmedbert-base-embeddings}/`. Each directory includes the full `hub-model-metadata.json` API response; neither contains weight files.

## Why these two

**E5-base-v2** provides the general English comparison. E5 was developed as a general sentence representation family using contrastive learning on weakly supervised text pairs (CCPairs), supporting retrieval, clustering, and classification. That objective is more directly relevant to frozen sentence features than predicting masked tokens. The family paper supplies the objective evidence; the exact v2 checkpoint's card supplies its extraction recipe. Neither establishes MRI image–report performance. Sources: [original E5 paper](https://arxiv.org/abs/2212.03533), [pinned E5 model card](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/README.md).

**NeuML PubMedBERT Embeddings** is an English biomedical sentence encoder built from Microsoft's PubMedBERT abstract/full-text model. The card documents sentence-transformers training with PubMed title–abstract and similar-title pairs, using `MultipleNegativesRankingLoss`, cosine similarity, scale 20, and one epoch. It is a medical-literature semantic model, not an MRI/radiology report specialist. Its published evaluation is text-only and cannot establish a cross-modal retrieval advantage on this project's reports. Choosing it tests whether medical sentence training helps without introducing a custom extraction stack. Source: [pinned NeuML model card](https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/b79526d6ef3645e0df4530322e266f24c829f5ef/README.md).

## Extraction and loader contract

| Property | E5-base-v2 | NeuML PubMedBERT Embeddings |
| --- | --- | --- |
| Backbone | Standard `BertModel`, 12 layers, width 768 | Standard `BertModel`, 12 layers, width 768 |
| Token window | 512 total tokens, including special tokens and prefix | 512 total tokens, including special tokens |
| Pooling | Attention-mask-aware mean of `last_hidden_state` | Attention-mask-aware mean of `last_hidden_state` |
| Text prefix | `query: ` for this feature-extraction experiment | Empty; no special prefix documented |
| Published sentence-transformers modules | Transformer → Pooling → Normalize | Transformer → Pooling |
| Learned downstream sentence projection in checkpoint | None in module graph | None in module graph |
| Loader | `AutoTokenizer` + `AutoModel`; standard BERT | `AutoTokenizer` + `AutoModel`; standard BERT |

Architecture/window sources: [E5 config](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/config.json), [NeuML config](https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/b79526d6ef3645e0df4530322e266f24c829f5ef/config.json). Sentence-pipeline sources: [E5 modules](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/modules.json), [E5 pooling](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/1_Pooling/config.json), [NeuML modules](https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/b79526d6ef3645e0df4530322e266f24c829f5ef/modules.json), [NeuML pooling](https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/b79526d6ef3645e0df4530322e266f24c829f5ef/1_Pooling/config.json).

E5 documents `query: ` for embeddings used as features or symmetric text similarity; asymmetric text retrieval uses separate `query: ` and `passage: ` roles. Here one report representation is reused for both image→text and text→image retrieval. Applying `query: ` consistently is the feature-extraction interpretation, not an upstream claim that E5 was trained for image retrieval. Audit lengths after inserting the prefix. NeuML examples pass unprefixed sentences. Sources: [E5 usage/FAQ](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/README.md), [NeuML Transformers example](https://huggingface.co/NeuML/pubmedbert-base-embeddings/blob/b79526d6ef3645e0df4530322e266f24c829f5ef/README.md).

The repository's [TextTower and ProjectionHead](../src/contrastive_learning/model.py) feed pooled features through `LayerNorm(768) → Linear(768,256) → GELU → Linear(256,256) → L2 normalization`. Selecting `mean` preserves both checkpoints' pooling and discards no trained sentence Dense head. It does bypass E5's final, parameter-free normalization: this screen deliberately uses **raw mean features before the existing task projection**, matching the current extraction approach. It must not be described as exact sentence-transformers E5 output. LayerNorm is not mathematically identical to upstream L2 normalization. NeuML's pre-projection mean features reproduce its declared sentence module sequence. Standard BERT's `pooler_output` is not part of either declared mean-pooling path.

Both owner cards show direct Transformers loading, and both inspected configs use ordinary `model_type: bert` with no `auto_map`. No `trust_remote_code=True`, txtai, sentence-transformers installation, or custom Python module is needed for this repo's extraction path. Preserve the existing locked Transformers/PyTorch environment and perform a small loader/forward smoke check after staging the pinned weights; metadata inspection alone is not a runtime test. Sources: the pinned model cards, configs, and module graphs above.

## Footprint and RTX 3070 8 GB assessment

The pinned API reports 109,482,240 float32 parameters for each model. E5 additionally lists an int64 buffer with 512 entries. Listed safetensors file sizes are **437,955,512 bytes** for E5 and **437,951,328 bytes** for NeuML, approximately 438 MB decimal / 418 MiB each. Those are remote metadata values, not locally downloaded or measured GPU allocations. Sources: the pinned Hub API responses linked above.

Derived weight-only estimates are approximately **0.408 GiB at float32** or **0.204 GiB if weights are explicitly stored at float16** per text encoder. Autocast alone does not guarantee half-sized parameter storage. The existing projection contains 264,192 trainable parameters at width 768→256→256, negligible next to the backbone's weights; this count is calculated from the local module definition. These estimates omit activations, attention workspaces, image backbone, optimizer state for other trainable modules, allocator overhead, and CUDA context.

Inference: both are plausible frozen text candidates on the project's RTX 3070 8 GB budget, individually and sequentially. The assessment is not an 8 GB fit guarantee for the full multimodal run. Long sequences and image batches can dominate temporary memory. Before the first full run, measure peak allocated/reserved CUDA memory on the frozen configuration and its longest actual batches; record dtype, batch size, token limit, and attention implementation. This research provides **no measured memory, throughput, or retrieval score** for either candidate.

## Earlier names and deferred alternatives

| Name encountered | Precise distinction and reason to defer in this screen |
| --- | --- |
| UCSD RadBERT | `UCSD-VA-health/RadBERT-RoBERTa-4m` is RoBERTa adapted to VA radiology reports; its card retains the older `zzxslp/...` example. Strong domain control, but masked-language-model adaptation does not establish sentence contrastive training. [Owner card](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m) |
| Stanford RadBERT | `StanfordAIMI/RadBERT` is a distinct BioBERT-initialized radiology language model, not an alias of UCSD's model. Keep for a later radiology adaptation comparison. [Owner card](https://huggingface.co/StanfordAIMI/RadBERT) |
| PubMedBERT / BiomedBERT | Microsoft's `BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` was renamed `BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`. That base biomedical language model is distinct from NeuML's sentence-trained derivative selected here. [Microsoft card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) |
| ClinicalBERT | The concrete earlier candidate is `emilyalsentzer/Bio_ClinicalBERT`, BioBERT adapted to MIMIC-III notes. ICU clinical prose and masked-language training answer a different question from this first sentence-encoder comparison; “ClinicalBERT” alone is insufficient identification. [Author card](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) |
| CXR-BERT-specialized / BioViL | `microsoft/BiomedVLP-CXR-BERT-specialized` has chest-X-ray multimodal contrastive training. Its documented inference requires custom code and `get_projected_text_embeddings`; plain `last_hidden_state` plus this repo's new head does not preserve that pretrained projected embedding. Defer until its trained projection and chest-domain transfer can be evaluated explicitly. [Microsoft card and inference example](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) |

The shortlist is a controlled test of general English versus biomedical sentence features, not a claim that either outperforms the recorded multilingual baseline. Freeze the rest of the experimental protocol before downloading weights or starting training, and select using the project's validation protocol rather than model-card leaderboards.
