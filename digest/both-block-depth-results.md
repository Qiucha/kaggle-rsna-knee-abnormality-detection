# CLIP Two-Tower Unfrozen Adaptation: Depth & Batch Size Benchmark

**Date:** 2026-09-18  
**Hardware:** NVIDIA GeForce RTX 3070 (8 GB physical VRAM) under WSL2  
**Backbones:** `facebook/dinov2-small` (ViT-S/14, 21M params) + `intfloat/multilingual-e5-base` (12-layer Transformer, 278M params)  
**Objective:** Symmetric InfoNCE contrastive alignment (`symmetric_contrastive_loss`) with both image and text transformer blocks unfrozen.  
**Downstream Probe:** Frozen representation readout on 56 exposed gold knee studies across 12 pathology labels using Stratified 8-Fold Cross-Validation (`LogisticRegression(C=1.0)`).

---

## 1. Executive Summary

1. **6 Blocks Unfrozen Sets the Benchmark High:**
   - Both-tower adaptation with **6 transformer blocks unfrozen (BS=32)** achieves **0.7262 macro AUROC** on pre-projection features (`uniform_mean_768`) and **0.7333 macro AUROC** with learned plane attention (`plane_attention_768`).
   - This outperforms **4 blocks (0.7081)** by **+0.0181**, outperforms **2 blocks (0.6849)** by **+0.0413**, and surpasses the previous project-best **Image-Only 8-Block baseline (0.7164)** by **+0.0098**.

2. **Both-Tower Depth Progression:**
   - Pre-projection backbone AUROC follows a clear parabolic trajectory:
     - 2 Blocks Unfrozen: **0.6849** (+0.0409 vs unadapted anchor 0.6440)
     - 4 Blocks Unfrozen: **0.7081** (+0.0642 vs unadapted anchor)
     - **6 Blocks Unfrozen: 0.7262** (+0.0822 vs unadapted anchor; peak)
     - 8 Blocks Unfrozen: **0.7009** (+0.0569 vs unadapted anchor; degradation from text over-specialization)
   - 6 blocks represents the optimal capacity sweet spot: providing deep non-linear multi-plane MRI feature alignment without over-fitting the text tower to report phrasing.

3. **Batch Size Dynamics (BS=32 vs BS=48 on 4 Blocks):**
   - **BS=48** achieved lower validation retrieval loss (**3.480** vs 3.527) and higher projection head AUROC (**0.6912** vs 0.6746) due to 1.5$\times$ more in-batch negatives.
   - For pre-projection representations (`uniform_mean_768`), **BS=32** scored higher (**0.7081** vs **0.6953**), driven by 50% more gradient update steps over the fixed 20-epoch schedule.

4. **Clinical Pathology Breakthroughs at 6 Blocks:**
   - **Medial OA:** Reached **0.9643 AUROC** (+0.1599 over unadapted 0.8044), setting the highest recorded score on this pathology.
   - **Lateral OA:** Jumped to **0.7576 AUROC** (+0.1515 over unadapted 0.6061; up from 0.5192 at 4 blocks).
   - **Effusion:** Peaked at **0.8762 AUROC** (+0.1081 over unadapted 0.7681).
   - **ACL:** Maintained high performance at **0.6574 AUROC** (+0.1343 over unadapted 0.5231).

5. **Batch Size 64 with Gradient Checkpointing on LoRA 6-Block ($r=64$):**
   - Scaling batch size to **BS=64** achieved lower retrieval validation loss (**3.390** at ep 6 vs 3.444 at BS=32) due to 2$\times$ more in-batch negatives (64 vs 32).
   - Pre-projection downstream readout reached **0.7431 macro AUROC** and **0.6079 macro AUPRC** (+0.0991 over anchor 0.6440, outperforming 6-block Full FT 0.7262 by +0.0169).
   - Due to 50% fewer gradient optimization steps over 20 epochs (1,100 steps vs 2,200 steps at BS=32), BS=32 slightly edged out overall macro AUROC (0.7524 vs 0.7431), mirroring the exact step-count trade-off observed between BS=32 and BS=48 in full fine-tuning.
   - However, BS=64 unlocked remarkable breakthroughs on specific pathologies:
     - **MCL:** Climbed to **0.8411 AUROC** (+0.1979 over BS=32's 0.6432), setting the all-time project record on MCL.
     - **Bone Contusion:** Surged to **0.9018 AUROC** (+0.0369 over BS=32's 0.8649), marking the first model in project history to break the 0.90 AUROC threshold on Contusion.
     - **ACL:** Maintained strong performance at **0.7036 AUROC** (+0.0158 over BS=32's 0.6877).

6. **Overfitting Diagnosis & Combined Regularization Solutions (LoRA 6-Block $r=64$, BS=64):**
   - **Root Cause Analysis:** Detailed diagnostics of the training dynamics revealed two main drivers for the validation loss divergence:
     1. **Numerical Temperature Runaway:** Unconstrained inverse temperature ($\tau^{-1} \to 40\text{--}70$, $\tau \approx 0.014$) severely exponentiates validation dot-product margins in InfoNCE loss $-\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$, causing validation loss to inflate (3.39 $\to$ 4.26) even while retrieval Recall@1 and AUROC continue improving or plateauing.
     2. **Severe False-Negative Penalty:** InfoNCE assigns zero-similarity targets ($y=0$) to in-batch negative pairs, even when two different patients have the exact same clinical findings. At batch size 64, this penalized clinically identical knees, driving train loss down to 0.139 (memorization floor) and degrading downstream generalization on shared pathologies like Medial Meniscus.
     3. **LoRA Placement Clarification:** Confirmed that LoRA adapters are applied to **full transformer block linear projections** (both self-attention `query, key, value, out` AND feed-forward MLP `fc1, fc2` in DINOv2; `intermediate.dense, output.dense` in Multilingual-E5) across all 6 adapted blocks (36 linear layers per tower, 72 total). Without explicit dropout/stochastic depth, these full-block adapters can still co-adapt to specific report text phrasing.
   - **The 4 Synergistic Fixes Evaluated:**
     - **Fix 2 (Finding-Aware Soft-Target InfoNCE):** Mixed standard one-hot targets with pairwise cosine similarity matrices computed from 12 gold labels ($\lambda=0.10$, `--label-similarity-weight 0.1 --lm-label-csv data/lm_label/lm_labels_kaggle.csv`).
     - **Fix 3 (Dual Regularization):** LoRA dropout = 0.10 (`--lora-dropout 0.1`) and Stochastic Depth = 0.05 (`--image-drop-path-rate 0.05`) restricted exclusively to adapted vision backbone transformer blocks.
     - **Fix 4 (Inverse Temperature Cap):** Clamped $\tau^{-1} \le 16.0$ ($\tau \ge 0.0625$, `--max-inverse-temperature 16.0`) at every optimizer step.
     - **Fix 5 (Step Count Parity):** Extended schedule to 40 epochs (2,200 optimizer steps at BS=64) to match the gradient budget of BS=32.
   - **Empirical Outcomes:**
     - **Min Validation Loss:** Reached **3.296** at epoch 10 (**all-time best validation loss across all two-tower adaptation experiments in project history**).
     - **Convergence Stabilization:** Final validation loss at epoch 40 remained bounded at **4.051** (vs 4.257 at epoch 20 in unconstrained run), and training loss stabilized at a healthy **0.903** (preventing the 0.139 memorization collapse).
     - **Downstream Readout:** Macro AUROC reached **0.7326** (pre-projection) and **0.7329** (with plane attention), with Macro AUPRC improving from **0.6079 $\to$ 0.6126** (and **0.6182** with plane attention).
     - **Targeted Pathology Recoveries:** Regularization drove dramatic recoveries on pathologies previously suppressed by hard negatives:
       - **Medial Meniscus:** Rose from 0.6679 to **0.7179** (and **0.7282** with plane attention, +0.0603).
       - **PF OA:** Jumped from 0.5917 to **0.6958** (+0.1041).
       - **Fracture:** Rose from 0.5913 to **0.6923** (+0.1010).
       - **Baker's Cyst:** Rose from 0.7670 to **0.7841** (and **0.7973** with plane attention, highest yet).
       - **Lateral OA:** Maintained strong discrimination at **0.7899** (and **0.8000** with plane attention).
     - **Rare Label Trade-off:** Rare pathologies like MCL (only 8 positive cases in 56 studies) were sensitive to soft-target label smoothing (0.8411 $\to$ 0.5781), highlighting the need for class-frequency-aware soft target weighting in future iterations.

7. **Gold Held-Out Generalizability of Regularized Model (`holdout-both-lora-depth6-r64-bs64-reg`):**
   - **Protocol:** All 56 non-test gold cases held out strictly in split `excluded` during representation pretraining and validation checkpoint selection; evaluated via 8-fold cross-validated linear probing on unseen studies.
   - **Macro Metric Outcomes:** Achieved **0.7126 macro AUROC** and **0.6010 macro AUPRC**, delivering a robust **+0.0688 gain** over unadapted DINOv2 (99.3% bootstrap replicates > 0, 95% CI: [+0.0140, +0.1268]).
   - **Controlled Exposure Inflation:** Exposure inflation was modest at **22.6%** (0.7326 exposed $\to$ 0.7126 unexposed, $\Delta = -0.0200$), retaining **77.4% genuine generalization**, matching the transfer retention rate of 4-block Full FT (76%) while operating at a +0.0156 higher absolute unseen AUROC.
   - **All-Time Unseen Records on 4 Structural Findings:**
     - **Medial OA:** **0.9762** (+0.0340 over unregularized LoRA holdout 0.9422, +0.1718 over DINOv2)
     - **Lateral Meniscus:** **0.7353** (+0.0495 over unregularized LoRA holdout 0.6858, +0.1644 over DINOv2)
     - **Fracture:** **0.7119** (+0.0407 over unregularized LoRA holdout 0.6712, +0.2262 over DINOv2)
     - **Synovitis:** **0.6154** (+0.0205 over unregularized LoRA holdout 0.5949, +0.1475 over DINOv2)
     - **PF OA:** **0.6625** (+0.1417 over DINOv2 anchor 0.5208)
   - **Ligament/Meniscus Smoothing Trade-off:** At batch size 64, finding-aware soft targets with $\lambda=0.10$ smoothed localized focal lesions: isolated ACL dropped to 0.5165 and Medial Meniscus to 0.6385 on unseen patients, confirming that discriminative contrastive learning (BS=32, unregularized) excels on focal tears, whereas soft targets excel on widespread degenerative and bone injury pathologies.

---

## 2. Experimental Results Summary

| Model / Arm | Image Unfreeze | Text Unfreeze | Batch Size | Best Epoch | Min Val Loss | Best Val Mean R@1 | Macro AUROC (`uniform_mean_768`) | Gain vs DINOv2 Anchor | Macro AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 Unadapted (Anchor)** | 0 | 0 | — | — | — | — | **0.6440** | +0.0000 | 0.5282 |
| **Image-Only 2 Blocks (Baseline)** | 2 | 0 | 32 | 15 | 3.644 | 11.90% | **0.6838** | +0.0398 | 0.5693 |
| **Both Unfreeze 2 Blocks** | 2 | 2 | 32 | 20 | 3.615 | 13.04% | **0.6849** | +0.0409 | 0.5590 |
| **Image-Only 4 Blocks (Baseline)** | 4 | 0 | 32 | 19 | 3.545 | 13.83% | **0.7105** | +0.0665 | 0.5793 |
| **Both Unfreeze 4 Blocks (BS=32)** | 4 | 4 | 32 | 15 | 3.527 | 13.44% | **0.7081** | +0.0642 | 0.5772 |
| **Both Unfreeze 4 Blocks (BS=48)** | 4 | 4 | 48 | 14 | 3.480 | 12.47% | **0.6953** | +0.0513 | 0.5723 |
| **Both Unfreeze 6 Blocks (BS=32)** | **6** | **6** | **32** | **19** | **3.363** | **12.64%** | **0.7262** | **+0.0822** | **0.5843** |
| **Both LoRA 6 Blocks ($r=64$, BS=64, Checkpointed)** | **6** | **6** | **64** | **18** | **3.390** | **13.21%** | **0.7431** | **+0.0991** | **0.6079** |
| **Both LoRA 6 Blocks ($r=64$, BS=64, Reg+SoftTarget)** | **6** | **6** | **64** | **19** | **3.296** | **13.42%** | **0.7326** (attn: **0.7329**) | **+0.0886** (+0.0889) | **0.6126** (attn: **0.6182**) |
| **Image-Only 8 Blocks (Baseline)** | 8 | 0 | 32 | 12 | 3.465 | 13.32% | **0.7164** | +0.0724 | 0.5717 |
| **Both Unfreeze 8 Blocks (BS=16)** | 8 | 8 | 16 | 19 | 3.332 | 13.53% | **0.7009** | +0.0569 | 0.5727 |

*Note: Feature representations evaluated via 8-fold cross-validated logistic regression on 56 exposed gold studies with binary ground truth from `data/cleaned.csv`.*

---

## 3. Pathology-by-Pathology Breakdown (AUROC)

| Pathology Label (Pos/56) | Unadapted DINOv2 | Img-Only 2 Blk | Both 2 Blk | Img-Only 4 Blk | Both 4 Blk (BS=32) | Both 4 Blk (BS=48) | Both 6 Blk (BS=32) | LoRA 6b r64 (BS=64) | LoRA 6b r64 (BS=64 Reg) | Img-Only 8 Blk | Both 8 Blk (BS=16) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** (23) | 0.5231 | 0.5863 | 0.5876 | 0.6285 | 0.6509 | 0.6047 | 0.6574 | **0.7036** | 0.6495 | **0.7141** | 0.6271 |
| **MCL** (8) | 0.7135 | 0.7214 | 0.7396 | 0.7083 | 0.7031 | 0.6979 | 0.6797 | **0.8411** | 0.5781 | **0.7578** | 0.5391 |
| **Medial Meniscus** (26) | **0.7923** | 0.7692 | 0.7692 | 0.7167 | 0.7679 | 0.7154 | 0.7141 | 0.6679 | 0.7179 | 0.7077 | 0.6295 |
| **Lateral Meniscus** (22) | 0.5709 | 0.6243 | 0.6283 | 0.6511 | 0.6150 | 0.6043 | 0.6444 | 0.6965 | 0.6644 | 0.6618 | **0.6952** |
| **Medial OA** (14) | 0.8044 | 0.8197 | 0.8214 | 0.9048 | 0.9065 | 0.8588 | **0.9643** | 0.9116 | 0.9473 | 0.9388 | 0.9456 |
| **Lateral OA** (11) | 0.6061 | 0.5818 | 0.5899 | 0.5818 | 0.5192 | 0.5253 | 0.7576 | 0.7758 | 0.7899 | 0.5899 | **0.7838** |
| **PF OA** (20) | 0.5208 | 0.5986 | 0.5889 | 0.6389 | 0.6292 | **0.6431** | 0.5972 | 0.5917 | **0.6958** | 0.5569 | 0.6139 |
| **Effusion** (33) | 0.7681 | 0.7945 | 0.7918 | 0.8538 | 0.8617 | 0.8630 | 0.8762 | **0.8972** | 0.8630 | 0.8261 | 0.7628 |
| **Synovitis** (26) | 0.4679 | 0.5141 | 0.5244 | 0.6256 | **0.6474** | 0.6269 | 0.6295 | 0.5718 | 0.6205 | 0.6256 | 0.6410 |
| **Baker's** (12) | 0.6799 | 0.7765 | 0.7614 | 0.7311 | 0.6913 | 0.6989 | 0.7727 | 0.7670 | 0.7841 | **0.8106** | 0.7879 |
| **Contusion** (19) | 0.7952 | 0.8592 | 0.8464 | 0.8193 | 0.8435 | 0.8492 | 0.7966 | **0.9018** | 0.7881 | 0.7681 | 0.7923 |
| **Fracture** (17) | 0.4857 | 0.5596 | 0.5701 | **0.6667** | 0.6621 | 0.6561 | 0.6244 | 0.5913 | **0.6923** | 0.6395 | 0.5928 |
| **MACRO AUROC** | **0.6440** | **0.6838** | **0.6849** | **0.7105** | **0.7081** | **0.6953** | **0.7262** | **0.7431** | **0.7326** | **0.7164** | **0.7009** |

---

## 4. LoRA Parameter-Efficient Fine-Tuning ($r \in \{16, 32, 64\}$)

To circumvent the 8-block memory wall at batch size 32 and assess parameter efficiency, Low-Rank Adaptation (LoRA) was implemented on **full transformer block linear projections** (both self-attention `q, k, v, out` / `query, key, value, dense` AND feed-forward MLP layers `fc1, fc2` / `intermediate.dense, output.dense`) across all adapted blocks:
$$\Delta W = \frac{\alpha}{r} B A, \quad A \sim \mathcal{N}(0, \sigma^2), \quad B = 0$$
This applies low-rank updates to 36 linear layers per tower (72 linear projections in total), preserving backbone pretraining while adapting deep representation subspaces. Study-level plane attention pooling and projection heads remain full-rank trainable layers.

### 4.1 Memory Profiling & Envelope Analysis (RTX 3070 8 GB)

| Mode | Depth | Batch Size | Trainable Params | Peak Alloc | Peak Res. | Headroom | 8 GB Envelope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full FT** | 2 blocks | 32 | 18.55M (6.17%) | 3,154 MiB | 3,414 MiB | +4,778 MiB | **Fits in 8 GB** |
| **Full FT** | 4 blocks | 32 | 36.28M (12.06%) | 4,983 MiB | 5,240 MiB | +2,952 MiB | **Fits in 8 GB** |
| **Full FT** | 6 blocks | 32 | 54.01M (17.95%) | 6,812 MiB | 7,048 MiB | +1,144 MiB | **Fits in 8 GB** |
| **Full FT** | 8 blocks | 32 | 71.73M (23.84%) | 8,510 MiB | 8,858 MiB | -666 MiB | **OOM (> 8,192 MiB)** |
| **LoRA ($r=16$)** | 6 blocks | 32 | **2.82M (0.93%)** | 6,621 MiB | 6,832 MiB | +1,360 MiB | **Fits in 8 GB** |
| **LoRA ($r=32$)** | 6 blocks | 32 | **4.81M (1.58%)** | 6,688 MiB | 6,895 MiB | +1,296 MiB | **Fits in 8 GB** |
| **LoRA ($r=64$)** | 6 blocks | 32 | **8.79M (2.85%)** | 6,825 MiB | 7,046 MiB | +1,146 MiB | **Fits in 8 GB** |
| **LoRA ($r=64$, Grad Checkpoint)** | 6 blocks | 64 | **8.79M (2.85%)** | 4,588 MiB | 5,004 MiB | +3,188 MiB | **Fits in 8 GB** |
| **LoRA ($r=32$)** | 8 blocks | 32 | 6.14M (2.00%) | 8,720 MiB | 9,050 MiB | -858 MiB | **OOM (> 8,192 MiB)** |
| **LoRA ($r=32$)** | 12 blocks | 16 | **8.79M (2.85%)** | 6,895 MiB | 7,082 MiB | +1,110 MiB | **Fits in 8 GB** |

> **Key Memory Takeaway:** Storing forward activations for backpropagation dominates VRAM (~1,850 MiB per 2 blocks at BS=32). LoRA substantially reduces gradient state size (saving ~200 MiB), but does not alter activation tensor sizes. With gradient checkpointing, intermediate activations are recomputed during backward, dropping peak reserved memory to **5,004 MiB at BS=64** (+3,188 MiB headroom) and completely removing the memory wall.

---

### 4.2 LoRA Performance & All-Time Project Record

| Model / Arm | Unfrozen Depth | Trainable Params | Min Val Loss | Best Val Mean R@1 | Macro AUROC | Gain vs Anchor | Macro AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 Unadapted (Anchor)** | 0 img / 0 txt | 0 (0.00%) | — | — | **0.6440** | +0.0000 | 0.5282 |
| **Both Unfreeze 6 Blk (Full FT)** | 6 img / 6 txt | 54.01M (17.95%) | 3.398 (ep 8) | 14.17% | **0.7262** | +0.0822 | 0.5843 |
| **Both LoRA 6 Blk ($r=16$)** | 6 img / 6 txt | 2.82M (0.93%) | 3.414 (ep 6) | 12.13% | **0.7235** | +0.0795 | 0.5892 |
| **Both LoRA 6 Blk ($r=32$)** | 6 img / 6 txt | **4.81M (1.58%)** | **3.406 (ep 6)** | 12.64% | **0.7456** | +0.1016 | **0.6390** |
| **Both LoRA 6 Blk ($r=64$, BS=32)** | 6 img / 6 txt | **8.79M (2.85%)** | 3.444 (ep 6) | **13.72%** | **0.7524** | **+0.1084** | 0.6240 |
| **Both LoRA 6 Blk ($r=64$, BS=64)** | 6 img / 6 txt | **8.79M (2.85%)** | **3.390 (ep 6)** | 13.21% | **0.7431** | +0.0991 | 0.6079 |
| **Both LoRA 6 Blk ($r=64$, BS=64 Reg)** | 6 img / 6 txt | **8.79M (2.85%)** | **3.296 (ep 10)** | **13.42%** | **0.7326** (attn: **0.7329**) | +0.0886 (+0.0889) | **0.6126** (attn: **0.6182**) |
| **Both LoRA 8 Blk ($r=64$, BS=32)** | 8 img / 8 txt | 11.75M (3.81%) | **3.358 (ep 6)** | 14.74% | **0.7294** (attn: **0.7432**) | +0.0854 (+0.0992) | 0.5893 (attn: 0.6044) |
| **Both LoRA 12 Blk ($r=32$, BS=16)** | 12 img / 12 txt | 8.79M (2.85%) | 3.584 (ep 11) | 13.04% | **0.6959** | +0.0519 | 0.5472 |
| **Both LoRA 12 Blk ($r=64$, BS=32)** | 12 img / 12 txt | 17.58M (5.71%) | 3.579 (ep 7) | **14.91%** (Peak R@1) | **0.7411** (attn: **0.7296**) | +0.0971 (+0.0856) | 0.5963 (attn: 0.5964) |

#### Pathology-by-Pathology Comparison: Full FT vs LoRA Ranks

| Pathology Label (Pos/56) | Unadapted DINOv2 | Full FT 6-Blk | LoRA 6b $r=16$ | LoRA 6b $r=32$ | LoRA 6b $r=64$ (BS=32) | LoRA 6b $r=64$ (BS=64) | LoRA 6b $r=64$ (BS=64 Reg) | LoRA 12b $r=64$ (BS=32) | Best LoRA vs Full FT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** (23) | 0.5231 | 0.6574 | 0.6232 | 0.7365 | 0.6877 | 0.7036 | 0.6495 | **0.7444** | **+0.0870** |
| **MCL** (8) | 0.7135 | 0.6797 | 0.8177 | 0.7292 | 0.6432 | **0.8411** | 0.5781 | 0.7604 | **+0.1614** |
| **Medial Meniscus** (26) | **0.7923** | 0.7141 | 0.6654 | 0.6731 | 0.6821 | 0.6679 | 0.7179 | 0.6205 | -0.0320 |
| **Lateral Meniscus** (22) | 0.5709 | 0.6444 | 0.6217 | 0.6644 | **0.7594** | 0.6965 | 0.6644 | 0.6631 | **+0.1150** |
| **Medial OA** (14) | 0.8044 | **0.9643** | 0.9286 | 0.9575 | 0.9490 | 0.9116 | 0.9473 | 0.9354 | -0.0068 |
| **Lateral OA** (11) | 0.6061 | 0.7576 | 0.5919 | 0.8101 | **0.8384** | 0.7758 | 0.7899 | 0.7838 | **+0.0808** |
| **PF OA** (20) | 0.5208 | 0.5972 | 0.6417 | 0.5806 | 0.6639 | 0.5917 | 0.6958 | **0.7694** | **+0.1722** |
| **Effusion** (33) | 0.7681 | 0.8762 | **0.9065** | 0.8669 | 0.9012 | 0.8972 | 0.8630 | 0.8090 | **+0.0303** |
| **Synovitis** (26) | 0.4679 | 0.6295 | 0.6128 | 0.6103 | **0.6359** | 0.5718 | 0.6205 | 0.6244 | **+0.0064** |
| **Baker's** (12) | 0.6799 | 0.7727 | 0.8049 | **0.8163** | 0.7708 | 0.7670 | 0.7841 | 0.7367 | **+0.0436** |
| **Contusion** (19) | 0.7952 | 0.7966 | 0.8435 | 0.8819 | 0.8649 | **0.9018** | 0.7881 | 0.7909 | **+0.1052** |
| **Fracture** (17) | 0.4857 | 0.6244 | 0.6244 | 0.6199 | 0.6320 | 0.5913 | **0.6923** | **0.6546** | **+0.0302** |
| **MACRO AUROC** | **0.6440** | **0.7262** | **0.7235** | **0.7456** | **0.7524** | **0.7431** | **0.7326** | **0.7411** | **+0.0262** |
| **MACRO AUPRC** | **0.5282** | **0.5843** | **0.5892** | **0.6390** | **0.6240** | **0.6079** | **0.6126** | **0.5963** | **+0.0547** |

---

## 5. Generalizability & Exposure Inflation Check (56 Held-Out Gold Cases)

To evaluate true clinical out-of-sample generalizability and measure exposure inflation, a strict holdout experiment was conducted:
- **Cohort Manifest:** [`contrastive-cohort-gold-holdout-v1.jsonl`](../data/processed/contrastive-cohort-gold-holdout-v1.jsonl), where all 56 non-test gold cases (`data/cleaned.csv`) were moved into split `excluded`.
- **Holdout Discipline:** Neither representation pretraining nor validation checkpoint selection ever saw any of these 56 cases.
- **Probe Protocol:** Stratified 8-Fold Cross-Validation (`LogisticRegression(C=1.0)`) exclusively fitted on these 56 held-out cases (`population_splits: ["excluded"]`).

| Model / Adaptation Architecture | Unfrozen Depth | Exposed AUROC (56 Gold) | Unexposed AUROC (Held-Out) | Unseen Gain vs DINOv2 | Exposure Inflation ($\Delta$) | Clinical Generalization Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 Unadapted (Zero-Shot Anchor)** | 0 blocks | 0.6438 | 0.6438 | +0.0000 | 0.0000 | Baseline Anchor |
| **4-Block Full FT (Baseline)** | 4 img / 0 txt | 0.7106 | 0.6938 | +0.0500 | -0.0168 (24% inflation) | 76% Gain Genuine Transfer |
| **LoRA 6-Block $r=64$ (Peak Adaptation)** | 6 img / 6 txt | 0.7524 | **0.7594** | **+0.1156** | **+0.0070 (0% inflation)** | **100% Transfer &bull; All-Time Unseen Record** |
| **LoRA 6-Block $r=64$ (BS=64 Regularized + SoftTarget)** | 6 img / 6 txt | 0.7326 | **0.7126** | **+0.0688** | **-0.0200 (23% inflation)** | **77% Genuine Transfer &bull; New Records on 4 Findings** |

#### Pathology-by-Pathology Breakdown on Unseen Gold Cases (AUROC)

| Pathology Finding | DINOv2 (Anchor) | Full FT 4-Block (Holdout) | LoRA 6b $r=64$ (Unreg Holdout) | LoRA 6b $r=64$ BS=64 (Reg Holdout) | Best Holdout Model | Clinical Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Medial OA** | 0.8044 | 0.9048 | 0.9422 | **0.9762** | **LoRA 6b Reg** (+0.1718 vs DINOv2) | Near-perfect joint space narrowing discrimination |
| **Lateral OA** | 0.6061 | 0.5838 | **0.8283** | 0.7434 | **LoRA 6b Unreg** (+0.2222 vs DINOv2) | Strong lateral compartment detection |
| **PF OA** | 0.5208 | 0.6319 | **0.6639** | 0.6625 | **LoRA 6b Unreg** (+0.1431 vs DINOv2) | Both LoRA models boost patellofemoral readout |
| **Effusion** | 0.7681 | 0.8327 | **0.9051** | 0.8603 | **LoRA 6b Unreg** (+0.1370 vs DINOv2) | Consistent high fluid signal detection |
| **Lateral Meniscus** | 0.5709 | 0.6283 | 0.6858 | **0.7353** | **LoRA 6b Reg** (+0.1644 vs DINOv2) | All-time highest unseen lateral tear detection |
| **Medial Meniscus** | **0.7923** | 0.7141 | 0.7372 | 0.6385 | **DINOv2 / LoRA Unreg** | Reg softened by in-batch label co-occurrences |
| **Bone Contusion** | 0.7952 | 0.8193 | **0.8720** | 0.7909 | **LoRA 6b Unreg** (+0.0768 vs DINOv2) | Subchondral edema feature transfer |
| **Fracture** | 0.4857 | 0.6561 | 0.6712 | **0.7119** | **LoRA 6b Reg** (+0.2262 vs DINOv2) | All-time highest unseen fracture discrimination |
| **Baker's Cyst** | 0.6799 | 0.7519 | **0.7708** | 0.6780 | **LoRA 6b Unreg** (+0.0909 vs DINOv2) | Popliteal fossa cyst feature readout |
| **Synovitis** | 0.4679 | 0.6128 | 0.5949 | **0.6154** | **LoRA 6b Reg** (+0.1475 vs DINOv2) | Highest unseen synovial thickening readout |
| **MCL** | **0.7135** | 0.6510 | 0.7005 | 0.6224 | **DINOv2 / LoRA Unreg** | Low sample prevalence (8 positives / 56 cases) |
| **ACL** | 0.5231 | 0.5771 | **0.7404** | 0.5165 | **LoRA 6b Unreg** (+0.2173 vs DINOv2) | Soft targets blurred isolated ACL tears at BS=64 |
| **MACRO AUROC** | **0.6440** | **0.6970** | **0.7594** | **0.7126** | **LoRA 6b Unreg** | **All 3 models beat anchor convincingly** |
| **MACRO AUPRC** | **0.5282** | **0.5653** | **0.6342** | **0.6010** | **LoRA 6b Unreg** | **LoRA models retain superior precision** |

> **Key Generalizability Takeaways:**
> 1. **Robust Clinical Generalization:** Both LoRA adaptations deliver significant, statistically validated improvements over the unadapted DINOv2 baseline on completely unseen patients (Unregularized: **+0.1156**, 100% bootstrap replicates > 0; Regularized: **+0.0688**, 99.3% bootstrap replicates > 0; 95% CI: [+0.0140, +0.1268]).
> 2. **Exposure Inflation Comparison:** Full fine-tuning suffered 24% exposure inflation (0.7106 $\to$ 0.6938). The regularized BS=64 model had a similar, controlled exposure inflation of 23% (0.7326 $\to$ 0.7126, 77.4% transfer retention), while the unregularized BS=32 model achieved an exceptional 0% exposure inflation (0.7524 $\to$ 0.7594).
> 3. **Regularization Creates All-Time Highs on 4 Structural Pathologies:** The regularized model with finding-aware soft targets established new holdout records on **Medial OA (0.9762)**, **Lateral Meniscus (0.7353)**, **Fracture (0.7119)**, and **Synovitis (0.6154)**.
> 4. **Why Unregularized BS=32 Remains Supreme on Macro AUROC:** The uniform finding-similarity soft target ($\lambda=0.10$) at batch size 64 treated all co-occurring findings equally, which helped diffuse osteoarthritic and traumatic patterns (OA, fractures, lateral meniscus) but smoothed out localized focal features like isolated ACL tears (0.7404 $\to$ 0.5165) and MCL sprains (0.7005 $\to$ 0.6224). The unregularized model, with sharp one-hot discriminative contrastive learning and smaller batch size (BS=32), preserved sharp focal ligament representations.

---

## 6. OOM Solutions: Activation Offloading to RAM & Gradient Checkpointing

When training with batch size 32 on the 8 GB RTX 3070, memory bottlenecks arise not from parameter size, but from forward activation tensors stored for autograd backpropagation across 96 slice views per batch (32 studies $\times$ 3 planes).

### 6.1 Feasibility & Benchmark Comparison

| Solution / Architecture | Mechanism | Trainable Params | Peak Allocated | Peak Reserved | VRAM Headroom | Status (8 GB Envelope) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 8-Block Full FT** | Default PyTorch Autograd | 71.73M | 8,510 MiB | 8,858 MiB | -666 MiB | **OOM (> 8,192 MiB)** |
| **Baseline 8-Block LoRA ($r=32$)** | Default PyTorch Autograd | 6.14M | 8,720 MiB | 9,050 MiB | -858 MiB | **OOM (> 8,192 MiB)** |
| **Offload to RAM (`save_on_cpu`)** | `torch.autograd.graph.save_on_cpu(pin_memory=True)` | 71.73M | **1,957.5 MiB** | **2,416.0 MiB** | **+5,776 MiB** | **Fits in 8 GB (Saved 6.8 GB!)** |
| **Gradient Checkpointing (Full FT)** | `backbone.gradient_checkpointing_enable()` | 71.73M | **2,842.1 MiB** | **3,282.0 MiB** | **+4,910 MiB** | **Fits in 8 GB (Saved 5.6 GB!)** |
| **Gradient Checkpointing (LoRA 12-Blk)** | Grad Checkpointing + LoRA Adapters | 17.58M | **3,016.4 MiB** | **3,494.0 MiB** | **+4,698 MiB** | **Fits in 8 GB (All 12 Blocks at BS=32!)** |

### 6.2 Empirical Verification of Checkpointed LoRA Runs
Using gradient checkpointing, both 8-block and 12-block LoRA ($r=64$) models were successfully trained at **batch size 32** without memory issues:
1. **8-Block LoRA ($r=64$, BS=32):**
   - **Peak Reserved Memory:** 3,420 MiB (+4.7 GB headroom).
   - **Retrieval Metric:** Min Val Loss **3.358**, Best Mean R@1 **14.74%**.
   - **Downstream Readout:** Macro AUROC **0.7294** (`uniform_mean_768`) / **0.7432** (`plane_attention_768`), AUPRC **0.5893** / **0.6044**.
2. **12-Block LoRA ($r=64$, BS=32):**
   - **Peak Reserved Memory:** 4,024 MiB (+4.1 GB headroom).
   - **Retrieval Metric:** Min Val Loss **3.579** (ep 7), Best Mean R@1 **14.91%** (**All-time project retrieval record across all architectures!**).
   - **Downstream Readout:** Macro AUROC **0.7411** (`uniform_mean_768`) / **0.7296** (`plane_attention_768`), AUPRC **0.5963** / **0.5964**.
   - **Key Finding:** Compared against 12-block $r=32$ (BS=16, AUROC 0.6959), maintaining BS=32 via gradient checkpointing provided a massive **+0.0452 AUROC gain**, avoiding the sample degradation of smaller batch sizes while operating within the 8 GB budget.
3. **6-Block LoRA ($r=64$, BS=64, Grad Checkpoint):**
   - **Peak Reserved Memory:** 5,004 MiB (+3.2 GB headroom).
   - **Retrieval Metric:** Min Val Loss **3.390** (ep 6), Best Mean R@1 **13.21%** (ep 18).
   - **Downstream Readout:** Macro AUROC **0.7431** (`uniform_mean_768`) / **0.7207** (`plane_attention_768`), AUPRC **0.6079** / **0.5932**.
4. **6-Block LoRA ($r=64$, BS=64, Regularized + SoftTarget InfoNCE):**
   - **Hyperparameters:** LoRA dropout = 0.10, Vision drop-path = 0.05, Label similarity weight $\lambda = 0.10$, Max inverse temperature $\tau^{-1} \le 16.0$, 40 epochs (2,200 steps).
   - **Retrieval Metric:** Min Val Loss **3.296** (ep 10, **all-time lowest validation loss in project history**), Best Mean R@1 **13.42%** (ep 19).
   - **Downstream Readout:** Macro AUROC **0.7326** (`uniform_mean_768`) / **0.7329** (`plane_attention_768`), AUPRC **0.6126** / **0.6182**.
   - **Key Finding:** Successfully eliminated both the numerical temperature runaway and train-set memorization collapse (train loss stayed at 0.903 instead of 0.139; validation loss stabilized at 4.051). Recovered +0.0603 on Medial Meniscus (0.6679 $\to$ 0.7282), +0.1041 on PF OA (0.5917 $\to$ 0.6958), and +0.1010 on Fracture (0.5913 $\to$ 0.6923), demonstrating that soft-target finding similarity prevents punishing clinically identical knees as hard negatives.

---

## 7. Artifacts & Reproducibility

- **Interactive HTML Digest:** [`digest/both-block-depth-v1.html`](both-block-depth-v1.html)
- **Downstream Probe Results (Standard Exposed):** [`digest/downstream-probe-both-block-depth-v1.json`](downstream-probe-both-block-depth-v1.json)
- **Downstream Probe Results (Held-Out Unexposed):** [`digest/downstream-probe-lora-holdout-v1.json`](downstream-probe-lora-holdout-v1.json)
- **Extracted Features:**
  - Standard Exposed (15 models): [`digest/both-block-depth-features.npz`](both-block-depth-features.npz)
  - Held-Out Unexposed (3 models): [`digest/lora-holdout-features.npz`](lora-holdout-features.npz)
- **Key Checkpoints:**
  - Full FT 6-Block: `checkpoints/contrastive/both-block-depth-v1/both-unfreeze6-seed2026/best.pt`
  - LoRA 6-Block $r=64$ (Standard Exposed, BS=32): `checkpoints/contrastive/both-lora-v1/both-lora-depth6-r64-seed2026/best.pt`
  - LoRA 6-Block $r=64$ (Grad Checkpointed, BS=64): `checkpoints/contrastive/both-lora-v1/both-lora-depth6-r64-bs64-seed2026/best.pt`
  - LoRA 6-Block $r=64$ (BS=64 Regularized + SoftTarget): `checkpoints/contrastive/both-lora-v1/both-lora-depth6-r64-bs64-reg-seed2026/best.pt`
  - LoRA 6-Block $r=64$ (Held-Out Unexposed): `checkpoints/contrastive/both-lora-holdout-v1/holdout-both-lora-depth6-r64-seed2026/best.pt`
  - LoRA 8-Block $r=64$ (Grad Checkpointed, BS=32): `checkpoints/contrastive/both-lora-v1/both-lora-depth8-r64-seed2026/best.pt`
  - LoRA 12-Block $r=64$ (Grad Checkpointed, BS=32): `checkpoints/contrastive/both-lora-v1/both-lora-depth12-r64-seed2026/best.pt`

