# Kaggle DICOM codecs

How the submission decodes compressed DICOM on Kaggle's offline image, and why each
part is there. Tickets: `.scratch/image-preproc-next/issues/01-vendor-dicom-codecs.md`
and `02-decode-counters-and-series-fallback.md`.

## The problem

The competition data page says hidden-test series come in "a mix of transfer syntaxes
(uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little
Endian)". Every file we have locally is uncompressed Explicit VR LE, so none of this
shows up in local runs.

Kaggle's offline image (Python 3.12.13, pydicom 3.0.2, numpy 2.0.2, Pillow 11.3.0) can
decode JPEG baseline, JPEG 2000 and RLE, but has **no decoder for JPEG Lossless
(`1.2.840.10008.1.2.4.57`, `.4.70`) or JPEG-LS (`.4.80`, `.4.81`)**. Evidence:

- `.scratch/input-preproc-hypotheses/r4b-kaggle-decode.md`: probe on the Kaggle image.
- `.scratch/input-preproc-hypotheses/r5-code-review.md` §F1: the failure-path trace.
- `.scratch/input-preproc-hypotheses/r7-codec-fix.md`: the prototype and patch sketch.
- `.scratch/input-preproc-hypotheses/r8-kaggle-codec.md`: the wheels checked on Kaggle.

Before this change, the pipeline failed like this (r5 §F1):

1. `_decode_pixels` swallowed every exception and returned `None`, with no log or counter.
2. If one or two of a plane's three selected slices failed, `read_slot` silently copied the
   nearest decoded slice into their channels.
3. If all three failed, `KneeReportDataset._load_images` raised, and the whole submission
   aborted with no CSV.

The 0.788 run finished, so no selected plane was wholly JPEG Lossless. Silent partial
substitution is still possible: about −0.005 LM-test macro AUROC per 10% of planes
affected (r5 F1-sim, corrected in r7). Any variant that changes series or slice
selection could meet a wholly JPEG-Lossless series and crash the scoring rerun.

## What changed

| File | Change |
|---|---|
| `kaggle/codecs/wheels/` | The four vendored wheels (below), 5.4 MiB. |
| `kaggle/codecs/fixtures/` | 6 lossless fixtures + `ref_{0,1}.npy`; 4 lossy fixtures; `manifest.json`. All synthetic. |
| `kaggle/codecs/make_fixtures.py` | Regenerates the synthetic fixtures and `manifest.json`. |
| `kaggle/codecs/install_and_selftest.py` | Offline install, then a fresh-process decode self-test. |
| `src/contrastive_learning/preprocessing.py` | Per-syntax decoder pins, decode counters, strict mode. |
| `src/contrastive_learning/data.py` | `rank_plane_candidates`, `StudyRecord.fallbacks`, fallback in `_load_images`. |
| `src/contrastive_learning/submission.py` | `submission_records` builds fallbacks; counters reset and reported. |
| `src/contrastive_learning/submission_training.py` | `copy_codecs` ships `kaggle/codecs` as `bundle/codecs` and hashes it. |
| `notebooks/kaggle_submission.ipynb` | Installs and self-tests the codecs before the prediction child. |
| `tests/test_dicom_codecs.py` | Pins, fixtures, counters, strict mode, fallback, bundle contents. |

`PREPROCESSING_VERSION` is **not** bumped, because no currently decodable file changes
pixels (see the parity evidence below).

### The Pillow pin is mandatory

Once pylibjpeg is installed, pydicom tries it **before** Pillow for JPEG baseline/extended
and JPEG 2000, and before its own decoder for RLE. `available_plugins` is a sorted set,
so it does not show this order. The r8 kernel showed the consequence on the real Kaggle
image: with the wheels installed, pydicom's default decoded the lossy JPEG-baseline
fixtures with pylibjpeg, and the pixels changed, including through the deployed bundle's
`_decode_pixels`. Pinning Pillow restored them bit for bit.

So `_decode_pixels` now pins a plugin per transfer syntax (`DECODING_PLUGINS`):

| Transfer syntax | Plugin |
|---|---|
| JPEG Baseline `.4.50`, JPEG Extended `.4.51` | `pillow` (as today) |
| JPEG 2000 Lossless `.4.90`, JPEG 2000 `.4.91` | `pillow` (as today) |
| RLE Lossless `1.2.840.10008.1.2.5` | `pydicom` (as today; pylibjpeg-rle would otherwise take over) |
| JPEG Lossless `.4.57`, SV1 `.4.70` | `pylibjpeg` (new; nothing decoded these before) |
| JPEG-LS `.4.80`, `.4.81` | `pylibjpeg` (new) |
| Uncompressed, HTJ2K, anything else | pydicom's default (no pin) |

The pin is applied as `ds.pixel_array_options(decoding_plugin=...)` on the dataset that
was already read, so the call path is otherwise unchanged. A pin whose plugin is not
installed is dropped (`decoding_plugin()` checks `get_decoder(uid).available_plugins`),
so an environment without the wheels behaves exactly as before.

If a pinned plugin raises, the default chain gets one more try, counted as
`pin_retry_decoded`. This cannot change a file that already decoded: before the wheels,
the pinned plugin was the only one available for that syntax, so a file it cannot decode
also failed before. The retry only rescues such files, for example 12-bit JPEG extended,
which Pillow cannot decode but libjpeg can.

## Rebuilding the wheels

The wheels are pinned. They are cp312 manylinux wheels for Kaggle's Python 3.12:

| Wheel | Size | sha256 (prefix) | Role |
|---|---|---|---|
| `pylibjpeg-2.1.0-py3-none-any.whl` | 25 KB | `25df9496` | Plugin router (entry points) |
| `pylibjpeg_libjpeg-2.4.0-cp312-…manylinux_2_17_x86_64.whl` | 3.4 MB | `01d950ef` | JPEG Lossless, JPEG-LS (and JPEG baseline, pinned away) |
| `pylibjpeg_openjpeg-2.6.0-cp312-…manylinux_2_28_x86_64.whl` | 2.0 MB | `aa6b0832` | JPEG 2000 (pinned away; kept for completeness) |
| `pylibjpeg_rle-2.2.0-cp312-…manylinux_2_17_x86_64.whl` | 272 KB | `1efa710f` | RLE (pinned away; kept for completeness) |

To fetch them again, from any machine with internet access:

```bash
pip download --only-binary=:all: --no-deps --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  -d kaggle/codecs/wheels \
  pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.4.0 pylibjpeg-openjpeg==2.6.0 pylibjpeg-rle==2.2.0
sha256sum kaggle/codecs/wheels/*.whl
```

If a version changes, rerun the self-test under CPython 3.12 with Kaggle's pins
(`pydicom==3.0.2 numpy==2.0.2 pillow==11.3.0`), then on Kaggle (see below). The install
uses `--no-deps` so it can never replace the image's numpy.

## The self-test

`bundle/codecs/install_and_selftest.py` (canonical copy: `kaggle/codecs/`):

1. Runs `pip install --no-index --no-deps --find-links wheels` for the four packages.
   It exits with code 2 if the install fails.
2. Starts a **fresh child interpreter**, because pydicom fixes its decoder plugins when it
   is first imported. The child checks:
   - JPEG Lossless (SV1 and `.57`), JPEG-LS and JPEG 2000 Lossless decoders are available;
   - the 6 lossless fixtures (JPEG-LL SV1 ×2, J2K ×2, JPEG-LS ×2; 12-bit unsigned and
     16-bit signed) decode bit-exactly to `ref_{0,1}.npy` with pydicom's default chain;
   - the pipeline's own `_decode_pixels` (imported from `--src`, or from `PYTHONPATH` in
     the `src.zip` layout) reproduces those references, and decodes the 4 lossy fixtures
     (JPEG baseline ×2, lossy J2K ×2) exactly as Pillow does. **This is the check that
     catches a missing pin.** Run against the deployed pre-pin bundle source with the
     wheels installed, it fails on both JPEG-baseline fixtures.
   - with Pillow 11.3.0 (Kaggle's), the Pillow decode of each lossy fixture matches the
     hash `make_fixtures.py` recorded without pylibjpeg installed (`manifest.json`).
3. Exits 0 only if everything passes. It prints a JSON report and `codec self-test PASSED`
   or `FAILED` on stderr.

`--skip-install` runs the self-test alone. `--require-pipeline` fails when the pipeline
cannot be imported, and the notebook always passes it.

The notebook runs the script with `subprocess.run(..., check=True)` **before** the
prediction child, so a failed install or self-test stops the notebook with an error
instead of producing a silently degraded submission. Nothing in the notebook kernel
imports pydicom; `torch` does not import it. On Kaggle, install plus self-test took
about 4 s in r8; the pipeline check adds a `torch` import to that.

There is still no JPEG Lossless `.4.57` (non-SV1) fixture, because nothing available
encodes that syntax. Its decoder is checked for availability only; libjpeg handles `.57`
and `.70` alike.

### Fixtures

All fixtures are **synthetic**: seeded phantoms (nested ellipses, a gradient, noise and a
fine periodic texture) in freshly built DICOM datasets with `PatientID` `SYNTHETIC`,
`PatientName` `SYNTHETIC^PHANTOM`, and UIDs generated deterministically from the fixture
name. No competition image or header is involved, and `tests/test_dicom_codecs.py`
asserts the `SYNTHETIC` markers on every `.dcm`.

| Files | Content | Encoder |
|---|---|---|
| `ref_0.npy`, `ref_1.npy` | 160² uint16 12-bit (with RescaleIntercept −1024), 160² int16 signed | numpy |
| `jpegll_{0,1}.dcm`, `j2kll_{0,1}.dcm`, `jpegls_{0,1}.dcm` | the refs, JPEG Lossless SV1 / J2K lossless / JPEG-LS lossless | GDCM 3.2.6 |
| `lossy_jpeg_baseline_{0,1}.dcm` | 256² 8-bit phantoms, quality 75 | Pillow 11.3.0 |
| `lossy_j2k_{0,1}.dcm` | 256² 12-bit-range phantoms, 16-bit irreversible, rate 20 | Pillow 11.3.0 |

On the lossy JPEG-baseline fixtures, Pillow and pylibjpeg disagree (max |Δ| = 1), so the
self-test can catch a missing pin; the lossy J2K fixtures decode identically under both
(both use OpenJPEG). A test asserts the JPEG-baseline disagreement when pylibjpeg is
installed.

To regenerate (byte-identical on rerun), use a throwaway CPython 3.12 venv with the
encoders and **without** pylibjpeg, since the script records Pillow hashes and refuses to
run with pylibjpeg installed:

```bash
python3.12 -m venv /tmp/fixgen
/tmp/fixgen/bin/pip install numpy==2.0.2 pydicom==3.0.2 pillow==11.3.0 python-gdcm==3.2.6
/tmp/fixgen/bin/python kaggle/codecs/make_fixtures.py
```

GDCM verifies each lossless file round-trips to its ref before the script writes the
manifest.

## Counters and strict mode

`preprocessing.decode_counts(reset=False)` returns `{transfer syntax: {event: count}}`
for the current process. Syntax names are pydicom keywords such as `JPEGLosslessSV1`
(`unknown` if the header is unreadable). The events are:

| Event | Meaning |
|---|---|
| `decoded` | A slice decoded. |
| `failed` | A slice did not decode (`_decode_pixels` returned `None`). |
| `pin_retry_decoded` | The pinned plugin failed and the default chain decoded the slice. |
| `substituted_slices` | A selected slice was replaced by its nearest decoded neighbour. |
| `undecodable_planes` | No selected slice of a series decoded (`read_slot` returned `None`). |
| `fallback_series` | Inference tried the next candidate series for a plane. |

`predict_submission` resets the counters once the records are built. At the end it
prints `decode_counts {...}` to stderr and stores them as `decode_counts` in
`submission.benchmark.json`. Counts are summed over ensemble members, because each
member re-reads every study. DataLoader workers keep their own counters, so training
totals are not collected.

Set `KNEE_STRICT_DECODE=1` for local validation. `read_slot` then raises `RuntimeError`
instead of substituting when any selected slice fails but at least one decodes. A plane
that fails entirely still returns `None`, as before, so `build_study_records` still
excludes it and inference can still fall back.

## Series fallback

`submission_records` (used by `predict_submission`) selects each plane's series with
`pick_study_slots`, exactly as training does, so **the first choice never changes**. It
also attaches `StudyRecord.fallbacks`, the rest of the plane's candidates in
`rank_plane_candidates` order:

1. the same preferred pool as selection (fluid-sensitive if the plane has any), by slice
   count and then manifest row;
2. then the plane's other series, as a last resort.

If a selected plane does not decode at all, `_load_images` tries the fallbacks in order.
It logs `decode fallback: study <uid> <plane>: <series> -> <series> (ok|failed)` to stderr
and counts `fallback_series`. It raises only when no candidate decodes. Pixels read from
a fallback series are never written to the disk cache, whose key names the selected
series. Training and manifest records carry no fallbacks, so they still fail loudly.

**Caveat.** A fallback series may be a sequence the model rarely saw in that slot, for
example a T1 series where every training study had a fluid-sensitive one. Its
predictions for that study are therefore less reliable. That is still strictly better
than aborting the whole submission, which scores nothing. Laterality is still resolved
from the selected series' headers, which are readable even when their pixels are not.

## Parity evidence

- **Local studies.** `read_slot` output, the `KneeReportDataset` tensors and the series
  selection are bit-identical between the deployed bundle's source and this change, on
  20 cohort studies (60 planes) and the 3 public example test studies. This holds both
  in the repo `.venv` (no pylibjpeg) and under CPython 3.12 with Kaggle's pins and the
  wheels installed. `rank_plane_candidates(...)[plane][0] == pick_study_slots(...)[plane]`
  on all 20.
- **Lossy decode.** With the wheels installed, the new `_decode_pixels` equals Pillow's
  decode on the lossy fixtures, while the deployed (unpinned) source differs on both
  JPEG-baseline fixtures, which is what r8 observed on Kaggle with its own lossy files.
- **Model outputs.** `predict_submission` on CPU with the deployed bundle
  (`lm-mlp-tower-size-58-dinov2-small-e5-base`) on the 3 example studies writes a
  byte-identical `submission.csv` with the old and the new source (sha256 `98fbcb1b…`).
  The new run reports `{"ExplicitVRLittleEndian": {"decoded": 27}}`.

## Verifying on Kaggle

The Kaggle-side check needs the user's approval (ticket 03). The template is the private
script kernel **`kyunhsuan/rsna-knee-codec-check-r8`** (source in
`.scratch/input-preproc-hypotheses/r8-kaggle-codec/kernel/`). It attaches the competition
data, a bundle, and the private wheel dataset `kyunhsuan/rsna-knee-codec-wheels-r8`, with
internet off. Its orchestrator never imports pydicom, and every probe runs in a fresh
child.

For a rebuilt bundle:

1. Upload the new `bundle/` as a private dataset. It now contains `codecs/`, so the
   separate wheel dataset is no longer needed.
2. Save a private version of `notebooks/kaggle_submission.ipynb` against it. In the log,
   check `codec self-test PASSED`, a JSON report with `"ok": true`, and the
   `decode_counts` line at the end of prediction. On the example studies it should show
   only `ExplicitVRLittleEndian` `decoded`.
3. Optionally, rerun the r8 kernel pointed at the new bundle. Its "after" pipeline hashes
   on the example studies must equal its "before" hashes, and its lossy-fixture
   `bundle_decode_pixels` must equal the pre-install values.
