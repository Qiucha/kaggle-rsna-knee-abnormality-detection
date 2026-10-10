# Ensemble selection: how to choose members

Updated 2026-10-09 22:40:54 +0800

All ensembles are per-label rank means of fine-tuned runs. LM-test is the criterion; gold-58 is the guard.

## Hand-picked combinations vs u9 alone

| Design | Combination | LM-test | Δ vs u9 [95% CI] | Gold-58 | Gold Δ vs u9 |
|---|---|---:|---|---:|---|
| single | u9 alone | 0.8834 | +0.0000 [+0.0000, +0.0000] | 0.8775 | +0.0000 [+0.0000, +0.0000] |
| sibling stack | u9 + clip | 0.8958 | +0.0124 [+0.0058, +0.0195] | 0.8659 | -0.0123 [-0.0287, +0.0029] |
| sibling stack | u9 + clip + 384 | 0.8975 | +0.0141 [+0.0066, +0.0222] | 0.8667 | -0.0116 [-0.0322, +0.0074] |
| sibling stack | u9 + clip + 384 + u6 | 0.8981 | +0.0147 [+0.0068, +0.0235] | 0.8712 | -0.0070 [-0.0258, +0.0100] |
| sibling stack | u9 + clip + 384 + u6 + run5 | 0.8976 | +0.0142 [+0.0059, +0.0231] | 0.8645 | -0.0137 [-0.0346, +0.0063] |
| backbone mix | u9 + coat | 0.8918 | +0.0085 [+0.0009, +0.0166] | 0.8722 | -0.0053 [-0.0223, +0.0105] |
| backbone mix | u9 + dS336 | 0.8926 | +0.0092 [+0.0023, +0.0164] | 0.8641 | -0.0137 [-0.0326, +0.0042] |
| backbone mix | u9 + coat + dS336 | 0.8970 | +0.0136 [+0.0061, +0.0220] | 0.8659 | -0.0117 [-0.0316, +0.0083] |
| strong core + other backbone | u9 + clip + 384 + coat | 0.9012 | +0.0178 [+0.0095, +0.0266] | 0.8711 | -0.0071 [-0.0263, +0.0116] |
| strong core + other backbone | u9 + clip + 384 + dS336 | 0.8999 | +0.0165 [+0.0092, +0.0244] | 0.8622 | -0.0160 [-0.0373, +0.0039] |
| strong core + other backbone | u9 + clip + 384 + softR4a | 0.9010 | +0.0175 [+0.0096, +0.0259] | 0.8638 | -0.0143 [-0.0342, +0.0044] |
| strong core + other backbone | u9 + clip + 384 + coat + dS336 | 0.9026 | +0.0192 [+0.0110, +0.0275] | 0.8674 | -0.0107 [-0.0316, +0.0093] |
| strong core + other backbone | u9 + clip + coat + dS336 + softR4a | 0.9033 | +0.0199 [+0.0118, +0.0287] | 0.8636 | -0.0143 [-0.0348, +0.0051] |
| input-size mix | u9 + 384 + omb224 | 0.8967 | +0.0132 [+0.0062, +0.0210] | 0.8718 | -0.0061 [-0.0254, +0.0128] |
| label-type mix | u9 + hard336 | 0.8879 | +0.0045 [-0.0032, +0.0120] | 0.8623 | -0.0152 [-0.0322, +0.0022] |
| label-type mix | u9 + clip + hard336 | 0.8974 | +0.0140 [+0.0062, +0.0220] | 0.8647 | -0.0133 [-0.0320, +0.0045] |
| seed average | run5 + run5' | 0.8824 | -0.0011 [-0.0107, +0.0083] | 0.8339 | -0.0443 [-0.0700, -0.0179] |
| seed average | u9 + run5 + run5' | 0.8915 | +0.0081 [+0.0017, +0.0146] | 0.8621 | -0.0159 [-0.0345, +0.0032] |
| previous candidate | softR4a + omb224 + run5 (built, unsubmitted) | 0.8948 | +0.0112 [+0.0015, +0.0212] | 0.8457 | -0.0323 [-0.0576, -0.0056] |
| previous candidate | softR4a + omb224 + u9 | 0.8963 | +0.0128 [+0.0055, +0.0206] | 0.8662 | -0.0113 [-0.0301, +0.0079] |

## Direct comparisons

| Question | LM-test Δ (A − B) | P(A > B) | Gold-58 Δ |
|---|---|---:|---|
| 5 members: diverse vs siblings | +0.0049 [+0.0007, +0.0097] | 0.99 | +0.0029 [-0.0084, +0.0138] |
| 5: diverse (no 384) vs siblings | +0.0056 [-0.0011, +0.0127] | 0.95 | -0.0010 [-0.0165, +0.0141] |
| 4th member: CoAtNet vs sibling u6 | +0.0031 [-0.0001, +0.0064] | 0.97 | -0.0000 [-0.0101, +0.0097] |
| 5 diverse vs 4 diverse | +0.0021 [-0.0038, +0.0079] | 0.76 | -0.0076 [-0.0202, +0.0047] |
| Two diverse 5-member sets | +0.0007 [-0.0035, +0.0050] | 0.63 | -0.0039 [-0.0129, +0.0058] |

## Selection rules, out of sample (k = 5, 5 split seeds x 200 half-splits)

| Split seed | Top-5 | Coverage | Greedy | Coverage − greedy |
|---|---:|---:|---:|---|
| 11 | 0.8950 | 0.8983 | 0.8992 | -0.0009 [-0.0094, +0.0052] |
| 22 | 0.8951 | 0.8986 | 0.8990 | -0.0004 [-0.0073, +0.0064] |
| 33 | 0.8942 | 0.8978 | 0.8983 | -0.0006 [-0.0079, +0.0065] |
| 44 | 0.8946 | 0.8983 | 0.8986 | -0.0002 [-0.0067, +0.0061] |
| 55 | 0.8951 | 0.8985 | 0.8991 | -0.0006 [-0.0086, +0.0065] |
| pooled | 0.8948 | 0.8983 | 0.8988 | -0.0005 [-0.0079, +0.0063] |

Per-label winners other than the best-macro run: edge +0.0274 on the selection half, +0.0024 held out (positive in 49%).

[Phase 2 digest page](site/pages/phase2.html#ensembles)
[Machine-readable snapshot](ensemble-selection-v1.json)
