"""Per-window features for every acquisition setting of each study, under two encoders.

The windowed rungs see one series per plane, chosen by ``pick_study_slots``: the
fat-suppressed one when the plane has any. The other weighting -- non-fat-suppressed,
about 42% of all series -- never reaches the model. This script asks whether it
carries label information the selected series do not, without retraining anything.

Each study gets six slots, ``primary`` (the manifest's own slot, what the rung sees)
and ``secondary`` (the best other series of the plane with the opposite fat-suppression
flag, ranked as ``rank_plane_candidates`` ranks them) for each plane. Windows are read
exactly as ``WindowDataset`` reads evaluation windows -- stratum centres, same crop,
size and band, right knees mirrored -- decoded once and encoded twice:

* ``finetuned``: the rung's own adapted backbone (``best.pt``). It was trained on the
  primary slots of the same training studies the heads train on, so its primary
  features are already fit to the targets and can crowd out anything new.
* ``pretrained``: the rung's starting backbone, which has seen neither slot. This is
  the clean test of whether the secondary series add information.

Output: one ``.npz`` shard per ``--shard-size`` studies under ``--output``, each with
``study_uids``, ``tokens_<encoder>`` ``[S, 6, K, D]`` float16, ``valid`` ``[S, 6, K]``
and ``fat_suppressed`` ``[S, 6]`` (-1 for an absent slot). Shards are fsynced, and a
rerun skips shards that load, so a dead session resumes rather than restarts.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning import finetune
from contrastive_learning.artifacts import read_json
from contrastive_learning.lm_proxy import load_test_population
from contrastive_learning.lm_supervision import load_population
from contrastive_learning.paths import host_data_root, host_series_root, series_csv
from contrastive_learning.preprocessing import (
    PLANES,
    apply_left_knee_windows,
    count_dicoms,
    read_windows,
)

SLOTS = [f"{role}/{plane}" for role in ("primary", "secondary") for plane in PLANES]
ENCODERS = ("finetuned", "pretrained")


def series_flags(series_csv: Path) -> tuple[dict, dict]:
    """``{study: [rows in manifest order]}`` and ``{series uid: fat-suppression bit}``."""
    by_study, flags = {}, {}
    with series_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_study.setdefault(row["StudyInstanceUID"], []).append(row)
            flags[row["SeriesInstanceUID"]] = int(row["Fat_Suppression"])
    return by_study, flags


def secondary_slots(record, rows, flags, series_root: Path) -> dict[str, Path | None]:
    """Per plane, the best series whose fat-suppression flag differs from the primary's.

    Ranked by slice count, then manifest row, as ``rank_plane_candidates`` ranks.
    """
    chosen = {}
    for plane in PLANES:
        primary = record.slots[plane]
        want = 1 - flags[primary.name]
        best, best_count = None, 0
        for row in rows:
            if row["Anatomical_Plane"] != plane or int(row["Fat_Suppression"]) != want:
                continue
            path = series_root / record.study_uid / row["SeriesInstanceUID"]
            count = count_dicoms(path)
            if count > best_count:  # Strict '>' lets the first row win a tie.
                best, best_count = path, count
        chosen[plane] = best
    return chosen


class SettingDataset(torch.utils.data.Dataset):
    """``[6, K, 3, H, W]`` windows, ``[6, K]`` validity and ``[6]`` flags per study.

    Primary windows come from the rung's evaluation window cache when present, else
    from the local mirror, else the source. Nothing is written to the window cache:
    at 336 px it would cost ~16 MB per study.
    """

    def __init__(self, records, config, by_study, flags, series_root):
        self.records, self.by_study, self.flags, self.series_root = records, by_study, flags, series_root
        self.reader = finetune.WindowDataset(records, config, training=False)
        self.kwargs = {k: v for k, v in finetune.image_kwargs(config).items() if k != "slices_per_plane"}
        self.windows = config["windows_per_plane"]

    def __len__(self):
        return len(self.records)

    def _plane(self, path, plane, laterality):
        read = read_windows(path, windows=self.windows, **self.kwargs, slice_rng=None)
        if read is None:
            return None
        return apply_left_knee_windows(*read, plane, laterality)

    def __getitem__(self, index):
        record = self.records[index]
        size = self.kwargs["image_size"]
        images = torch.zeros(len(SLOTS), self.windows, 3, size, size)
        valid = torch.zeros(len(SLOTS), self.windows, dtype=torch.bool)
        flags = torch.full((len(SLOTS),), -1, dtype=torch.int8)
        cached = None
        cache = self.reader.cache_path(record)
        if cache.exists():
            try:
                cached = torch.load(cache, weights_only=True)
            except Exception:  # a damaged entry is derived data; read the source instead
                cached = None
        secondary = secondary_slots(record, self.by_study[record.study_uid], self.flags, self.series_root)
        for i, plane in enumerate(PLANES):
            flags[i] = self.flags[record.slots[plane].name]
            if cached is not None:
                images[i], valid[i] = cached["images"][i].float(), cached["valid"][i]
            else:
                read = self._plane(self.reader._series(record.slots[plane]), plane, record.laterality)
                if read is not None:
                    images[i], valid[i] = read
            path = secondary[plane]
            if path is not None:
                read = self._plane(path, plane, record.laterality)
                if read is not None:
                    j = len(PLANES) + i
                    images[j], valid[j], flags[j] = *read, self.flags[path.name]
        return {"images": images, "valid": valid, "flags": flags, "study_uid": record.study_uid,
                "from_cache": cached is not None}


def load_encoders(config, checkpoint: Path, device):
    """The rung's adapted backbone and its pretrained starting point, both in eval mode."""
    encoders = {}
    for name in ENCODERS:
        torch.manual_seed(config["seed"])
        backbone, _ = finetune.load_backbone(config)
        model = finetune.WindowModel(backbone, config["head_hidden"], config["dropout"], config["pooling"],
                                     window_groups=config.get("window_groups"))
        if name == "finetuned":
            state = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
            loaded = model.load_state_dict(state["trainable_state_dict"], strict=False)
            if loaded.unexpected_keys:
                raise ValueError(f"Unexpected keys in best.pt: {loaded.unexpected_keys[:5]}")
        encoders[name] = model.requires_grad_(False).eval().to(device)
    return encoders


def write_shard(path: Path, arrays: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def shard_complete(path: Path, uids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as shard:
            return shard["study_uids"].tolist() == uids and all(f"tokens_{e}" in shard for e in ENCODERS)
    except Exception:  # truncated by a session death: rebuild
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A windowed rung's output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit", type=int, help="Smoke test: only this many studies per split")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = read_json(args.checkpoint / "config.json")
    finetune.validate_config(config)
    groups, *_ = load_population(config, finetune.ROOT)
    test_rows = load_test_population(finetune.ROOT / config["cohort_manifest"],
                                     finetune.ROOT / config["lm_labels_csv"],
                                     finetune.ROOT / config["gold_labels_csv"])[0]
    splits = {"train": groups["train"], "validation": groups["validation"],
              "lm_test": test_rows, "gold": groups["gold"]}
    if args.limit:
        splits = {name: rows[: args.limit] for name, rows in splits.items()}
    records = [r for rows in splits.values() for r in rows]
    by_study, flags = series_flags(Path(series_csv(host_data_root())))
    series_root = Path(host_series_root())

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "splits.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "slots": SLOTS, "encoders": list(ENCODERS),
         "splits": {name: [r.study_uid for r in rows] for name, rows in splits.items()}}, indent=1))

    encoders = load_encoders(config, args.checkpoint, args.device)
    shards = [records[i : i + args.shard_size] for i in range(0, len(records), args.shard_size)]
    started = time.perf_counter()
    for number, shard in enumerate(shards):
        path = args.output / f"shard-{number:04d}.npz"
        uids = [r.study_uid for r in shard]
        if shard_complete(path, uids):
            continue
        dataset = SettingDataset(shard, config, by_study, flags, series_root)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
        out = {"study_uids": [], "valid": [], "fat_suppressed": [], "from_cache": [],
               **{f"tokens_{e}": [] for e in ENCODERS}}
        for item in loader:
            images, valid = item["images"].to(args.device)[None], item["valid"].to(args.device)[None]
            with torch.autocast(device_type=args.device.split(":")[0], enabled=config["amp"]):
                for name, model in encoders.items():
                    out[f"tokens_{name}"].append(model.tokens(images, valid)[0].half().cpu().numpy())
            out["study_uids"].append(item["study_uid"])
            out["valid"].append(item["valid"].numpy())
            out["fat_suppressed"].append(item["flags"].numpy())
            out["from_cache"].append(item["from_cache"])
        write_shard(path, {k: np.asarray(v) if k == "study_uids" else np.stack(v) for k, v in out.items()})
        done = min((number + 1) * args.shard_size, len(records))
        print(json.dumps({"shard": number, "studies": f"{done}/{len(records)}",
                          "secondary_present": int((np.stack(out["fat_suppressed"])[:, len(PLANES):] >= 0).sum()),
                          "elapsed_min": round((time.perf_counter() - started) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
