# Next image and text tower candidates

Status: **candidates 1, 2, 4 and 5 run (2026-09-25); candidate 3 (DINOv3) not run, gated**. Recorded 2026-09-24. Results are in [next-model-pairs-v1.md](next-model-pairs-v1.md): the larger text towers (E5-large, BGE-M3 at 512 tokens) lead on every AUROC but no gain excludes zero, 1,024 tokens and register tokens did not help, and RAD-DINO is worse. Start from the completed [four-pair tower comparison](tower-size-factorial-v1.md) and [58-gold LM-head follow-up](lm-head-tower-size-58-v1.md).

**Follow-up (2026-09-27):** [codec-declip-v1.md](codec-declip-v1.md) retrains the reference on the codec-fix pipeline, which gives a noise-level replicate, and adds DeCLIP on DINOv2-small + E5-base, RAD-DINO + E5-base, and RAD-DINO + BGE-M3-512.

The current reference is **DINOv2-small + multilingual E5-base**. Its frozen image encoder with a three-seed LM-supervised MLP head scored **0.7659 macro AUROC on all 58 gold studies**. All four prior one-tower comparisons have paired intervals spanning zero. The next runs should test a specific representation or report-length hypothesis, rather than assume closer tower parameter counts will help.

| Order | Pair to test against the reference | Question | Preparation for the run |
| --- | --- | --- | --- |
| 1 | [`facebook/dinov2-with-registers-small`](https://huggingface.co/facebook/dinov2-with-registers-small) + E5-base | Do register-trained visual features improve knee finding readout at roughly the same image-tower size? | Exclude the four register tokens when averaging patch tokens in training, downstream feature extraction, and exported inference. Verify LoRA block discovery and checkpoint reload. |
| 2 | DINOv2-small + [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | Does a different multilingual text representation help, and do longer reports matter? | First cap at **512 tokens** to isolate the encoder. Then run **1,024 tokens** as a separate context-length test. BGE-M3 does not require E5's `query: ` instruction. Check memory and keep effective batch size comparable. |
| 3 | [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) + E5-base | Does newer visual pretraining improve the image tower? | Model access is gated. The ViT emits four register tokens; exclude them from patch pooling. Confirm the backbone, LoRA, and image processor work with the local pipeline before a full run. |
| 4 | DINOv2-small + [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | Does more text capacity within E5 help? | Retain E5 mean pooling, `query: ` prefix, and 512-token cap. Profile memory on the 8 GB GPU; use gradient accumulation if the physical batch must shrink. |
| 5 | [`microsoft/rad-dino`](https://huggingface.co/microsoft/rad-dino) + E5-base | Does medical-image pretraining transfer to this task? | RAD-DINO was trained on **chest X-rays**, so knee MRI transfer is uncertain. Check its image resolution and preprocessing before comparing with DINOv2. |

## Comparison protocol

1. Change **one tower at a time** from the DINOv2-small + E5-base reference. Pin exact model revisions, preprocessing, tokenizer behavior, and any code adapter in each run config.
2. Use the same gold-holdout cohort and report view. Preserve the prior contrastive checkpoint-selection rule (validation mean Recall@1); record retrieval and loss curves. Run multiple contrastive seeds where feasible because the earlier size comparison used one seed per pairing.
3. Fit the same LM-supervised 12-label head on `data/lm_label/lm_labels_kaggle.csv`, with all **58 gold studies excluded** from head training and validation. Select head epochs on LM-label validation loss, then report macro and per-label AUROC on the same 58 gold studies. The [58-case protocol and artifacts](lm-head-tower-size-58-v1.md) give the current reference.
4. Treat gold differences as **exploratory**: the 58 labels have already been examined repeatedly, including the two formerly reserved test cases. Report paired study intervals and seed variation, and distinguish retrieval gains from pathology-label gains.

The [report-length audit](report-length-statistics.json) found that **14.2%** of original reports exceeded the current 512-token E5 limit. This motivates BGE-M3's separate 1,024-token test; it does not predict an AUROC gain. The image models' published feature claims and the text models' retrieval claims come from their linked model cards, not from knee-MRI experiments.
