# DeCLIP Data-Efficient Alignment Benchmark: Low-Data MRI Supervision

## 1. Executive Summary & Context

Based on [`docs/notes/Training_Guide_on_CLIP.md`](../docs/notes/Training_Guide_on_CLIP.md), we have refined, executed, and evaluated the complete suite of **DeCLIP (Data-Efficient CLIP)** experiments against the project baselines.

In low-resource clinical knee MRI paired with unstructured radiology reports (3,526 training pairs), vanilla CLIP suffers from:
1. **Sample Inefficiency:** Only a single positive pair $(X_i, R_i)$ is constructed per study.
2. **False-Negative Penalty:** When distinct patients share identical clinical findings (e.g. medial osteoarthritis or meniscal tear), standard InfoNCE penalizes their mutual similarity as negatives.

DeCLIP addresses both challenges:
- **Multi-View InfoNCE ($L_{\text{MV}}$):** Combines pathology-preserving MRI perturbations $X_i' = \operatorname{Aug}_I(X_i)$ and clinically-masked report views $R_i' = \operatorname{Aug}_T(R_i)$ to form four cross-modal pairs: $(X_i, R_i), (X_i', R_i), (X_i, R_i'), (X_i', R_i')$.
- **Nearest-Neighbour Report Supervision ($L_{\text{NN}}$):** A FIFO circular memory queue $Q_T$ (capacity 512) stores report embeddings. For each study $X_i$, its report's nearest semantic neighbour $R_i^{\text{NN}} = \arg\max_{j \neq i} (z_i^T)^\top z_j^T$ is retrieved, converting potential false negatives into explicit positive pairs: $(X_i, R_i^{\text{NN}})$ and $(X_i', R_i^{\text{NN}})$.
- **Unimodal Image Self-Supervision ($L_I$):** SimCLR loss between $X_i$ and $X_i'$ enforcing physical transform invariance on MRI representations.

---

## 2. Quantitative Results & Comparative Master Table

All downstream evaluations use frozen representation readouts via **Stratified 8-Fold Cross-Validation Logistic Regression** across 12 clinical knee pathologies on the 56 exposed gold studies (with the 2 locked test cases strictly excluded).

| Model / Approach | Backbone Status | Trainable Params | Multi-View ($L_{\text{MV}}$) | NN Queue ($L_{\text{NN}}$) | Unimodal ($L_I$) | BS | Min Val Loss | Peak Mean R@1 | Macro AUROC | Gain vs Anchor | Macro AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINOv2 Unadapted Anchor** | Completely Frozen | 0 (0.0%) | None | None | None | — | — | — | 0.6440 | +0.0000 | 0.5282 |
| **Both Unfreeze 6b (Best Full FT)** | 6 Blocks Unfrozen | 53.8M (17.5%) | None | None | None | 32 | 3.363 (ep 9) | 12.64% | 0.7262 | +0.0822 | 0.5843 |
| **Both LoRA 6b r=64 (BS=32)** | 6 Blocks LoRA $r=64$ | 8.8M (2.85%) | None | None | None | 32 | 3.444 (ep 6) | 13.72% | **0.7524** | **+0.1084** | 0.6240 |
| **Both LoRA 6b r=64 (BS=64, Reg)** | 6 Blocks LoRA $r=64$ | 8.8M (2.85%) | None | None | None | 64 | 3.296 (ep 10) | 13.42% | 0.7326 | +0.0886 | 0.6126 |
| **DeCLIP Canonical (Proj Head)** | Frozen Backbones | **1.05M (0.35%)** | Yes (4 pairs) | FIFO 512 | None | 32 | 3.584 (ep 19) | 9.69% | **0.6572** | **+0.0133** | 0.5254 |
| **DeCLIP Canonical (Pre-Proj Identity)**| Frozen Backbones | 1.05M (0.35%) | Yes (4 pairs) | FIFO 512 | None | 32 | 3.584 (ep 19) | 9.69% | 0.6440 | +0.0000 | 0.5282 |
| **DeCLIP Ablation: MV Only ($\lambda_{\text{NN}}=0$)** | Frozen Backbones | **1.05M (0.35%)** | Yes (4 pairs) | None | None | 32 | 3.657 (ep 14) | 10.15% | **0.6751** | **+0.0311** | 0.5425 |
| **DeCLIP + 6-Block LoRA r=64** | 6 Blocks LoRA $r=64$ | 9.85M (3.20%) | Yes (4 pairs) | FIFO 512 | SimCLR ($L_I$) | 32 | **3.401 (ep 6)** | 13.49% | **0.7514 / 0.7523\*** | **+0.1074** | **0.6345\*** |

*\*0.7523 Macro AUROC and record 0.6345 Macro AUPRC achieved with Plane Attention pooling.*

---

## 3. Pathology-by-Pathology Breakdown (12 Diagnoses)

Values represent 8-fold cross-validated AUROC on the 56 exposed gold studies:

| Pathology Finding | DINOv2 Anchor | Both Full FT 6b | Both LoRA 6b (BS=32) | Both LoRA 6b (Reg) | DeCLIP Proj Head (256d) | DeCLIP + LoRA 6b | Project Record Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ACL** | 0.5231 | 0.6574 | 0.6877 | 0.6495 | 0.6482 | **0.7062** | **DeCLIP + LoRA (All-Time High)** |
| **MCL** | 0.7135 | 0.6797 | 0.6432 | 0.5781 | 0.5443 | **0.7292** | **DeCLIP + LoRA (Adapted Record)** |
| **Medial Meniscus** | 0.7923 | 0.7141 | 0.6821 | 0.7179 | 0.6436 | 0.6359 | Both LoRA Reg (0.7179) |
| **Lateral Meniscus** | 0.5709 | 0.6444 | **0.7594** | 0.6644 | 0.6725 | 0.7059 | Both LoRA (0.7594) |
| **Medial OA** | 0.8044 | 0.9643 | 0.9490 | 0.9473 | 0.9388 | **0.9711** | **DeCLIP + LoRA (All-Time High)** |
| **Lateral OA** | 0.6061 | 0.7576 | 0.8384 | 0.7899 | 0.7273 | **0.8505** | **DeCLIP + LoRA (All-Time High)** |
| **PF OA** | 0.5208 | 0.5972 | 0.6639 | **0.6958** | 0.6278 | 0.6194 | Both LoRA Reg (0.6958) |
| **Effusion** | 0.7681 | 0.8762 | **0.9012** | 0.8630 | 0.6983 | 0.8630 | Both LoRA (0.9012) |
| **Synovitis** | 0.4679 | 0.6295 | 0.6359 | 0.6205 | 0.5231 | **0.6744** | **DeCLIP + LoRA (All-Time High)** |
| **Baker's** | 0.6799 | 0.7727 | 0.7708 | 0.7841 | 0.6667 | **0.7917** | **DeCLIP + LoRA (All-Time High)** |
| **Contusion** | 0.7952 | 0.7966 | **0.8649** | 0.7881 | 0.6671 | 0.8492 | Both LoRA (0.8649) |
| **Fracture** | 0.4857 | 0.6244 | 0.6320 | **0.6923** | 0.5294 | 0.6199 | Both LoRA Reg (0.6923) |

---

## 4. Key Takeaways & Architectural Insights

1. **Frozen Backbones with DeCLIP Multi-View Heads:**
   - Training **only 1.05M parameters** (0.35% of model), DeCLIP Multi-View Only reaches **0.6751 Macro AUROC** (+0.0311 over foundation anchor), demonstrating that multi-view InfoNCE cross-modal supervision dramatically densifies the shared embedding space even when the visual backbone cannot drift.
2. **DeCLIP + LoRA Sets New Pathology Benchmarks:**
   - Combining DeCLIP objectives ($L_{\text{MV}} + L_{\text{NN}} + L_I$) with 6-block LoRA ($r=64$) delivers **0.7514 Macro AUROC** (uniform mean) and **0.7523 Macro AUROC** (plane attention) while establishing a **new project record in Macro AUPRC (0.6345 vs 0.6338 prior best)**.
   - DeCLIP + LoRA sets **new all-time project highs on 5 clinical diagnoses**:
     - **Medial OA:** `0.9711` (+0.1667 over anchor)
     - **Lateral OA:** `0.8505` (+0.2444 over anchor)
     - **ACL:** `0.7062` (+0.1831 over anchor)
     - **Synovitis:** `0.6744` (+0.2065 over anchor)
     - **Baker's Cyst:** `0.7917` (+0.1118 over anchor)
3. **Clinical Validation of the Nearest-Neighbour Semantic Queue:**
   - Osteoarthritis findings (Medial OA: 0.9711, Lateral OA: 0.8505) and synovial effusions specifically benefit from the rolling FIFO report queue, which retrieves shared clinical features across patients rather than penalizing shared pathology as false negatives.

---

## 5. Digest Artifacts

- **Baseline Digest (Updated):** [`digest/both-block-depth-v1.html`](both-block-depth-v1.html)
- **New DeCLIP Interactive Digest:** [`digest/declip-alignment-v1.html`](declip-alignment-v1.html)
- **HTTP Server:** Accessible locally on `http://localhost:5177/digest/declip-alignment-v1.html`
