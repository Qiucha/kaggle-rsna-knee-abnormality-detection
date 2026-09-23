# Both-tower block adaptation

The [matched one-block matrix](../configs/contrastive/claude-translated-dfd07db-both-block-v1.json)
compares adapting the final image block alone with adapting the final image and text
blocks together at seeds 2026, 2027, and 2028. The text-only matrix checks whether
moving the text tower can affect a frozen image encoder. The
[depth matrix](../configs/contrastive/both-block-depth-v1.json) explores two, four,
six, and eight adapted blocks per tower on the original-language cohort.

On the 56 eligible gold studies, adding the text block to the image-block arm
changed twelve-label macro AUROC by +0.0005, −0.0001, and −0.0030 across the matched
seeds. Each paired interval included zero. Image-only adaptation improved over
unadapted DINOv2 by roughly +0.024 to +0.027; the text-only arm left DINOv2 weights
identical. The measured one-block result therefore gives no evidence that adapting
the text tower improves the image representation's pathology readout.

The separate depth sweep found its highest recorded pre-projection macro AUROC at
six unfrozen blocks per tower (0.7262), compared with 0.6849, 0.7081, and 0.7009
at two, four, and eight blocks. These are development readouts: the gold studies
were exposed to representation training or checkpoint selection, so they do not
estimate performance on independent patients. LoRA experiments and submission
selection use separate, gold-holdout runs.
