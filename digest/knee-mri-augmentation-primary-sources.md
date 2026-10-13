# Knee MRI augmentation: primary-source check

Checked 2026-10-12. This note supplies external context for a local slice-range and augmentation ablation; it does not establish optimal settings for Novo.

## MRNet precedent

The original MRNet paper describes training-time rotations in **[-25°, 25°]**, shifts in **[-25, 25] pixels**, and **50% horizontal flipping**. Images were standardized to 256 × 256, so the translation bound corresponds to about 9.8% of image width. These are historical settings, not evidence that stronger augmentation will improve our model. The paper reports a combined training recipe rather than an isolated comparison of augmentation strengths. [Bien et al., Methods: preprocessing and MRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)

MRNet accepts a variable-length MRI series and pools features across its slices. The described method does not specify a central normalized slice restriction such as [0.2, 0.8]. Its architecture and aggregation differ from a fixed-budget slice sampler; broader coverage in Novo therefore needs its own comparison. [Bien et al., Methods: MRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)

The paper's histogram intensity standardization is **preprocessing**, fitted on training exams and applied to all splits. It should not be counted as stochastic intensity augmentation. [Bien et al., Methods: preprocessing](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)

## Spatial consistency and intensity candidates

TorchIO's first-party augmentation tutorial applies spatial transforms consistently to images grouped in a subject. For a model that consumes ordered slices, a reasonable implementation choice is to sample one in-plane transform per study/view and reuse it across its slices. This is an engineering inference from coherent volume augmentation, not an MRNet ablation result or a requirement to reuse identical 2D transforms across differently oriented views. [TorchIO augmentation tutorial source](https://github.com/TorchIO-project/torchio/blob/main/docs/tutorials/augmentation.md)

TorchIO's official documentation distinguishes spatial augmentation from intensity augmentation and lists gamma contrast, noise, blur, bias-field, and MRI artifact transforms. This establishes available MRI-oriented transform families, not their benefit for our task. [TorchIO augmentation reference](https://docs.torchio.org/latest/transforms/augmentation/)

## Implications for our experiment

- Establish the actual **supervised** baseline before attributing effects to stronger augmentation; an augmentation helper existing elsewhere is insufficient evidence that fine-tuning uses it.
- First compare slice coverage at a fixed slice count and augmentation recipe, and augmentation at fixed coverage. Widening coverage with the same count also spreads sampled positions farther apart, so it changes spatial sampling density.
- Reuse the existing conservative geometric and gain recipe as an initial augmentation candidate. Keep flips disabled while the project relies on its normalized left-knee convention. MRNet's horizontal-flip precedent does not resolve our orientation assumptions.
- Treat any extra intensity family as a subsequent independent comparison. A study-wide gain or gamma parameter avoids introducing arbitrary slice-to-slice exposure changes; voxel noise need not be identical across slices. This is a proposed design choice, not a measured optimum.
- Choose settings using the validation split, paired seeds, and the same primary metric. Preserve a held-out test split for final evaluation.

No training runs or performance measurements were performed for this source check.
