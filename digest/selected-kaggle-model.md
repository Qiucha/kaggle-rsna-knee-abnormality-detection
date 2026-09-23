# Selected Kaggle model

The current submission uses the `best.pt` checkpoint from
`holdout-both-lora-depth6-r64-seed2026`, followed by twelve pathology classifiers
fitted separately on gold labels. Its first public leaderboard score was **0.717**
(submission `56479080`).

| Component | Setting |
| --- | --- |
| Image encoder | Pretrained `facebook/dinov2-small` |
| Report encoder | Pretrained `intfloat/multilingual-e5-base`; mean pooling, `query: ` prefix, 512-token limit |
| Adaptation | LoRA in the **last 6 transformer blocks of each tower**; backbone weights frozen |
| LoRA | Rank **64**, alpha **128**, dropout **0** |
| Contrastive training | **3,478** paired MRI studies and reports; **433** validation studies; symmetric image–report contrastive loss; no pathology labels |
| Training schedule | **20 epochs**, batch size **32**, AdamW, seed **2026**; `best.pt` selected at epoch **17** by validation retrieval |
| Learning rates | Image and text adapters **1e-4**; projection layers **1e-3**; weight decay **0.02**; 10% warmup then cosine decay |
| Other settings | 256-dimensional contrastive projections; mixed precision; no gradient checkpointing or label-similarity soft targets |

The 56 usable gold-labeled studies were held out of contrastive training and
checkpoint selection. Afterward, the image encoder was frozen and **12 separate
logistic-regression heads** (`C=1.0`) were fitted on those 56 studies, one per
competition finding. Eight-fold out-of-fold evaluation gave **0.759 macro ROC-AUC**;
the final heads were refitted on all 56 studies. The Kaggle inference bundle uses
image backbone features and these heads. It does not run the report encoder or the
contrastive projection layers.

Sources: [holdout experiment matrix](../configs/contrastive/both-lora-holdout-v1.json),
[competition bundle configuration](../configs/kaggle/lora-holdout-v1.json), and
[benchmark and leaderboard record](kaggle-submission-pipeline.md).
