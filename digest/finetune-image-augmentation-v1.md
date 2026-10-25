# Image augmentation in supervised fine-tuning

**Status (2026-10-13): not adopted.** Image augmentation on the A30 recipe gives no gain. A30 (0.8917 LM-test,
0.890 public) remains the best single model.

Code: branch `feat/label-image-augmentation`. Comparison: `scripts/analysis/compare_image_augmentation.py`, which writes
`finetune-image-augmentation-v1.json` in the same format as the lateral-rebalance comparison.

## Trial: A30 + image augmentation

Run `checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30`, launched 2026-10-12 11:01 and
finished 2026-10-13 00:11 (Taipei), 13.2 h.

The trial copies the 30-epoch lateral oversample + noise arm (`...-lat-os-noise-ep30`, "A30") and changes only the
run name and the image augmentation. A30's recipe:

- OmniRad-base, nine unfrozen blocks, batch size 1 with accumulation 16, K8 at 336 px.
- Soft targets, seed 2026, the [0.2, 0.8] slice band.
- One extra draw of every study observed positive for Lateral Meniscus, Lateral OA or Synovitis, with the other
  labels masked on extra draws.
- N(0, 0.1) noise on the lateral targets.
- Cosine over 30 epochs, patience 30 (off), and a divergence stop at validation BCE 0.02 above its best for 3 epochs.

The trial starts from the pretrained weights, with no `resume_from`. Checkpoint selection remains validation masked BCE.

## Result

| | LM-test | Δ vs A30 [95% CI] | Gold-58 | Δ vs A30 [95% CI] |
|---|---|---|---|---|
| A30 + augmentation | 0.8889 | −0.0028 [−0.0111, +0.0055] | 0.8648 | −0.0166 [−0.0380, +0.0050] |
| A30 | 0.8917 | — | 0.8814 | — |
| u9-b1 | 0.8834 | | 0.8775 | |
| Noise floor: run 5 seed 2027 − 2026 | | −0.0029 [−0.0110, +0.0061] | | +0.0066 [−0.0153, +0.0291] |

Against u9-b1, augmentation scores +0.0055 [−0.0064, +0.0184] LM-test and −0.0126 [−0.0350, +0.0101] gold-58,
inside the −0.024 veto.

- Neither interval against A30 excludes zero, and both point estimates sit inside the seed floor, with gold leaning
  negative.
- Training barely changed. Both runs used all 30 epochs (the divergence stop never fired) and both select epoch 22.
  Validation BCE is 0.4212 for the trial against A30's 0.4199. The final training BCE, measured on augmented inputs,
  is 0.3175 against A30's 0.3149. The train–validation gap is unchanged, so these magnitudes did not regularise.
- Per label (Δ vs A30), the larger moves are losses: MCL (−0.031 LM-test, −0.054 gold) and Effusion on gold
  (−0.092). ACL rises on LM-test (+0.018). The three lateral targets do not move. On LM-test, Lateral Meniscus
  moves −0.0026 [−0.0290, +0.0239], Lateral OA −0.0008 and Synovitis +0.0103.
- Cost: 26.3 min per epoch against A30's 26.0, with 5.4 GiB peak memory.
- Caveats: one seed per run, and only one augmentation strength was tried. Stronger or different transforms (flips,
  elastic deformation, slice jitter) remain untested.

## Configuration and behavior

Adding an `image_augmentation` object to a fine-tuning config turns on the transforms during supervised training:

```json
"image_augmentation": {
  "rotation_degrees": 8.0,
  "zoom": 0.08,
  "shift": 0.05,
  "intensity": 0.10
}
```

Omit the object to keep the existing unaugmented recipe. An empty object turns on the defaults above. A partial
object overrides only the parameters it specifies. Unknown keys and nonnumeric, nonfinite or out-of-range values fail
before the output directory is created. Rotation accepts [0, 180] degrees. The other magnitudes accept [0, 1].

The transforms reuse the contrastive helper:

- Rotation: ±8°.
- Zoom-in: 1.00–1.08.
- Translation: ±0.05 in normalized grid coordinates, about ±2.5% of image extent before zoom.
- Intensity: multiplicative gain 0.90–1.10, clamped to [0, 1].

Resampling is bilinear with border padding. There are no stochastic flips.

Each study and plane draws a fresh transform on every read, oversampled reads included. The plane's three channels
and all K windows share that draw, and different planes draw independently. Invalid windows stay zero, masks pass
through unchanged, and raw dataset and cache tensors are never augmented in place.

Augmentation runs on the training device, before model normalization and outside autocast. Validation, LM-test, gold
scoring and exported inference do not apply these transforms. An independent device generator seeded with `seed + 2`
leaves the sampling, dropout and target-noise random streams untouched. `last.pt` saves the augmentation RNG state.

## Reproduce

From the repository root, with the GPU environment:

```bash
python scripts/training/finetune_supervised.py \
  --config configs/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30.json \
  --output checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30
python scripts/analysis/compare_image_augmentation.py --output digest/finetune-image-augmentation-v1.json
python scripts/digests/build_digest_site.py
```

- Inputs not in git: `data_root` (and optionally `train_series_mirror`) in `paths.json`, the cohort manifest and LM
  labels CSV named in the config, and, for the comparison, the finished run directories (`predictions.npz`,
  `report.json`, `config.json`) of run 5 (both seeds), u9-b1 and A30 under `checkpoints/finetune/`.
- To try another strength, copy the config and change `name` and the `image_augmentation` magnitudes; then point
  the `AUG` entry of `RUNS` in `scripts/analysis/compare_image_augmentation.py` at the new run directory.

## Validation

77 tests passed across fine-tuning, preprocessing, submission, DeCLIP and image-cache tests. A CUDA smoke check on
cached MRI windows at `[1, 3, 8, 3, 336, 336]` passed the finite/range, mask-preservation and invalid-window checks.

Sources: [`finetune-image-augmentation-v1.json`](finetune-image-augmentation-v1.json), run dir
`checkpoints/finetune/r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30` (keeps `last.pt`).
