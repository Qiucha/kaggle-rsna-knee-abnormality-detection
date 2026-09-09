# Image tensor shape audit

The model passes `[B,3,3,224,224]` to DINOv2 as **`[3B,3,224,224]`**, never `[3B,224,224]`. The axes are study, plane, slice/channel, height, width. Planes are Sagittal, Coronal, Axial.

| Operation | Before | After | Meaning |
| --- | --- | --- | --- |
| Stack selected slices | 3 × `[H,W]` | `[3,H,W]` | Slices become channels; no averaging |
| Resize (temporary batch axis) | `[1,3,H,W]` | `[1,3,224,224]` | Spatial interpolation only |
| Stack planes, then studies | `[3,224,224]` per plane | `[B,3,3,224,224]` | Preserve plane and channel order |
| Augmentation flatten / restore | `[B,3,3,H,W]` | `[3B,3,H,W]` → original | Shared spatial transform within each plane's channels |
| ImageTower reshape | `[B,3,3,224,224]` | `[3B,3,224,224]` | Row `b*3+p`, channel `s` |
| Channel normalization | `[3B,3,224,224]` | same | Per-channel affine transform |
| DINOv2 patch convolution | `[3B,3,224,224]` | `[3B,384,16,16]` | Learned mixing across all three slices and each 14×14 patch |
| Flatten spatial grid, transpose | `[3B,384,16,16]` | `[3B,256,384]` | Slice/channel no longer a separate axis |
| Add CLS, transformer | `[3B,256,384]` | `[3B,257,384]` | Each plane remains a separate encoder input |
| Mean patch tokens | `[3B,256,384]` | `[3B,384]` | Reduce patch axis only |
| Concatenate CLS and patch mean | 2 × `[3B,384]` | `[3B,768]` | Both summaries retained |
| Restore studies and planes | `[3B,768]` | `[B,3,768]` | Add plane embeddings |
| Attention scores / softmax | `[B,3,768]` | `[B,3]` | Normalize weights over planes |
| Weighted plane aggregation | `[B,3]`, `[B,3,768]` | `[B,768]` | Deliberately reduce plane axis |
| Projection, L2 normalization | `[B,768]` | `[B,256]` | One embedding per study |

All nine **slice slots** enter the encoder, as three images with three channels each. They are not nine independent grayscale image encodings. No averaging or slice reduction occurs before DINOv2. Batch/plane identity is recoverable after flattening using `flat[b*3+p,s]`; slice identity remains the channel axis until the patch convolution mixes channels.

Preprocessing can affect source identity: short series repeat selected indices; failed decodes use the nearest successfully decoded selected slice; mismatched decoded dimensions become zeros. A fully undecodable plane raises in dataset loading. Right-knee sagittal orientation reverses slice-channel order intentionally; coronal/axial orientation flips spatial width. Thus preservation of nine tensor slots does not establish nine distinct, successfully decoded DICOM slices for every real study.

## Logging and reproduction

Enable tracing on any training/evaluation command with `CONTRASTIVE_SHAPE_TRACE=1`; redirect stderr to a JSONL file. This logs every PyTorch operation before and after, including backbone internals, reshape/view, flatten's component operations, permute/transpose, indexing, convolution, and reductions. It also records NumPy slice stacking and percentile aggregation. Scope and sequence pair the before/after records; named `image.*` events explain axis semantics. Tracing is off by default and includes no tensor values. It is verbose: enable for a bounded diagnostic run. Existing running processes must load the updated code to use it.

Offline reproduction (no downloads):

```sh
.venv/bin/python scripts/audit_image_shapes.py 2> digest/image-shape-trace.jsonl
.venv/bin/python -m pytest tests/test_image_shape_trace.py tests/test_preprocessing.py -q
```

The saved trace uses the full 12-block, 384-wide DINOv2-small architecture with random weights and two synthetic studies at 224×224. All 18 distinct slice markers are recovered at the backbone input; embeddings have shape `[2,256]`. This verifies computation, not a trained checkpoint's quality or real DICOM provenance.

Regression tests independently verify every batch/plane/channel value (including noncontiguous input), correct study-level pooling, rejection of missing axes, complete before/after operation pairing, and nonzero influence from all nine channels with no cross-study influence using a smaller real DINOv2 model.
