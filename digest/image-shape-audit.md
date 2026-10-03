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

> **Correction, 2026-09-10.** This section previously described a tracing harness —
> `src/contrastive_learning/shape_trace.py`, `scripts/audit_image_shapes.py`,
> `tests/test_image_shape_trace.py`, and a `CONTRASTIVE_SHAPE_TRACE=1` switch — and gave
> commands to regenerate the trace. **None of those files exist**, in the working tree or
> anywhere in git history (`git log --all -- <path>` returns nothing for each; no source file
> mentions `CONTRASTIVE_SHAPE_TRACE`). The commands could never have run as written. The
> section below states what is actually present.

The saved trace `digest/image-shape-trace.jsonl` **is** committed (added in `62a2a8f`) and is
the evidence behind the table above. It holds 1,176 records: 585 matched `before`/`after`
operation pairs, plus four named `image.*` events (`image.input`, `image.flatten`,
`image.dino_input`, `image.dino_output`) that label axis semantics. Each record carries
`scope`, `sequence`, `operation`, and tensor shapes/dtypes — **no tensor values**. Those counts
are verifiable directly from the file.

It was produced against the full 12-block, 384-wide DINOv2-small architecture with random
weights and two synthetic studies at 224×224. All 18 distinct slice markers are recovered at the
backbone input; embeddings have shape `[2,256]`. That verifies computation, not a trained
checkpoint's quality or real DICOM provenance.

**The trace cannot currently be regenerated.** The code that produced it was never committed.
Treat the JSONL as a frozen historical artifact: cite it, but do not advertise a reproduction
command. Rebuilding the harness is optional work, not a prerequisite for trusting the table —
the reshape and pooling claims are readable directly from `model.py:136-155`, and the input
shape guard those claims depend on lives at `model.py:135`.

What *is* runnable today is the regression coverage, in **`tests/test_image_tower.py`**:

```sh
.venv/bin/python -m pytest tests/test_image_tower.py tests/test_preprocessing.py -q
```

It independently verifies every batch/plane/channel value reaching the backbone (including
noncontiguous input), correct study-level pooling under uniform plane attention, rejection of
missing plane or channel axes, and nonzero gradient influence from all nine slice channels with
no cross-study leakage, using a small real `Dinov2Model`. That is the substance of the
regression claim; only the trace-pairing check is absent, since it lived in the missing file.
