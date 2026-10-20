"""Install the vendored DICOM codec wheels offline, then prove they decode, in a fresh process.

Usage:  python install_and_selftest.py [--wheels DIR] [--fixtures DIR] [--src DIR]
                                       [--require-pipeline] [--skip-install]
Exit 0 = pass, 1 = a decode check failed, 2 = the wheel install failed.

Checks (see fixtures/manifest.json and kaggle/codecs/README.md):
* JPEG Lossless (SV1 and .57), JPEG-LS and JPEG 2000 lossless decoders are available.
* Every lossless fixture decodes bit-exactly to its ref_<k>.npy with pydicom's default
  decoder chain.
* When the pipeline source is importable (``--src``, or the inherited PYTHONPATH), its
  ``_decode_pixels`` reproduces the reference for every fixture, and decodes the lossy
  JPEG baseline and JPEG 2000 fixtures exactly as Pillow does. That last check is what
  catches a missing decoder pin: once pylibjpeg is installed, pydicom's default decodes
  lossy JPEG baseline with it instead of Pillow, and the pixels differ.
* With the reference Pillow version (11.3.0, Kaggle's), the Pillow decode of each lossy
  fixture also matches the hash make_fixtures.py recorded without pylibjpeg installed.

All fixtures are synthetic phantoms with synthetic headers (make_fixtures.py).

Why a fresh process: pydicom resolves decoder plugins (pylibjpeg entry points) when
``pydicom.pixels.decoders`` is first imported. Installing into a process that has already
imported pydicom leaves JPEG-Lossless unavailable, so the check runs in a child
interpreter, exactly as the Kaggle notebook's prediction child would.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle"]

CHILD = r"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pydicom, PIL
from pydicom.pixels import pixel_array
from pydicom.pixels.decoders import (JPEGLosslessSV1Decoder, JPEGLosslessDecoder,
                                     JPEG2000LosslessDecoder, JPEGLSLosslessDecoder)
fixtures, src, require_pipeline = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
manifest = json.loads((fixtures / "manifest.json").read_text())
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
report = {"python": sys.version.split()[0], "pydicom": pydicom.__version__,
          "numpy": np.__version__, "pillow": PIL.__version__, "decoders": {}, "files": {}}
failures = []
for d in (JPEGLosslessSV1Decoder, JPEGLosslessDecoder, JPEG2000LosslessDecoder, JPEGLSLosslessDecoder):
    report["decoders"][d.UID.keyword] = sorted(d.available_plugins)
    if not d.is_available:
        failures.append(f"no decoder for {d.UID.keyword}")

pp = None
if src:
    sys.path.insert(0, src)
try:
    from contrastive_learning import preprocessing as pp
    report["pipeline"] = str(Path(pp.__file__).parent)
except Exception as error:
    report["pipeline"] = f"not importable: {type(error).__name__}: {error}"
    if require_pipeline:
        failures.append("pipeline source not importable")

def pipeline_check(path, ds, array, row):
    # _decode_pixels must equal the reference pixels, rescaled exactly as it does.
    if pp is None:
        return
    got = pp._decode_pixels(path)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    want = array.astype(np.float32) * slope + intercept
    pin = getattr(pp, "decoding_plugin", lambda _: "")  # absent in pre-pin pipelines
    row["pipeline_plugin"] = pin(str(ds.file_meta.TransferSyntaxUID)) or "default"
    row["pipeline_exact"] = got is not None and got.dtype == want.dtype and np.array_equal(got, want)
    if not row["pipeline_exact"]:
        failures.append(f"{path.name}: pipeline decode differs")

for name, ref_name in sorted(manifest["lossless"].items()):
    path, row = fixtures / name, {}
    report["files"][name] = row
    try:
        ref = np.load(fixtures / ref_name)
        ds = pydicom.dcmread(path)
        row["ts"] = ds.file_meta.TransferSyntaxUID.keyword
        got = ds.pixel_array
        row["exact"] = got.shape == ref.shape and np.array_equal(got.astype(np.int64), ref.astype(np.int64))
        if not row["exact"]:
            failures.append(f"{name}: not bit-exact")
        pipeline_check(path, ds, ref, row)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {str(error)[:160]}"
        failures.append(f"{name}: {row['error']}")

lossy = manifest["lossy"]
reference_pillow = PIL.__version__ == lossy["pillow_version"]
for name, expected in sorted(lossy["files"].items()):
    path, row = fixtures / name, {}
    report["files"][name] = row
    try:
        ds = pydicom.dcmread(path)
        row["ts"] = ds.file_meta.TransferSyntaxUID.keyword
        pillow = pixel_array(path, decoding_plugin="pillow")
        # Informational: whether pydicom's unpinned default would change these pixels.
        row["default_equals_pillow"] = bool(np.array_equal(pixel_array(path), pillow))
        if reference_pillow:
            row["pillow_matches_reference"] = sha(pillow) == expected["pillow_sha256"]
            if not row["pillow_matches_reference"]:
                failures.append(f"{name}: Pillow decode differs from the recorded reference")
        pipeline_check(path, ds, pillow, row)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {str(error)[:160]}"
        failures.append(f"{name}: {row['error']}")

report["failures"] = failures
report["ok"] = not failures
print(json.dumps(report, indent=1))
sys.exit(0 if report["ok"] else 1)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheels", type=Path, default=HERE / "wheels")
    parser.add_argument("--fixtures", type=Path, default=HERE / "fixtures")
    parser.add_argument(
        "--src", type=Path, default=None,
        help="pipeline source root holding contrastive_learning/ (default: ../src if present)",
    )
    parser.add_argument(
        "--require-pipeline", action="store_true",
        help="fail unless the pipeline's _decode_pixels can be imported and checked",
    )
    parser.add_argument("--skip-install", action="store_true", help="self-test only")
    args = parser.parse_args()
    src = args.src if args.src is not None else HERE.parent / "src"
    if not args.skip_install:
        # --no-deps: never let the resolver touch the image's numpy (Kaggle: 2.0.2).
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
             "--find-links", str(args.wheels), *PACKAGES],
            check=False,
        )
        if install.returncode != 0:
            print("codec wheel install FAILED", file=sys.stderr)
            return 2
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(args.fixtures),
         str(src) if src.is_dir() else "", "1" if args.require_pipeline else "0"],
        check=False,
    )
    print("codec self-test", "PASSED" if result.returncode == 0 else "FAILED", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
