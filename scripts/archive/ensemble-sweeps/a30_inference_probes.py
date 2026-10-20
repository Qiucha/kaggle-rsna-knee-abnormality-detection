"""Inference-only probes on A30 (2026-10-14): checkpoint averaging, weight soups, window-jitter TTA.

Run from the t3code-d7aef637 worktree (the code that trained A30) with .venv-cuda:
    python a30_inference_probes.py --output checkpoints/finetune/a30-inference-probes
Each variant writes <name>.npz with the same keys as a run's predictions.npz, so the
comparison script can treat it as a member. Centre-window variants read the eval window
cache; jitter passes decode DICOMs with one stratum-uniform draw per window, the
training-time jitter, from a generator seeded by (pass, study).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

W = Path("/home/q/.t3/worktrees/novo-project/t3code-d7aef637")
sys.path.insert(0, str(W / "src"))
from contrastive_learning import finetune as ft  # noqa: E402
from contrastive_learning.lm_proxy import load_test_population  # noqa: E402

FT = W / "checkpoints/finetune"
A30 = "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-ep30"
AUG = "r4a-k8-omnirad-base-soft-336-u9-b1-lat-os-noise-aug-ep30"
U9 = "r4a-k8-omnirad-base-soft-336-u9-b1"


class JitterDataset(ft.WindowDataset):
    def __init__(self, records, config, pass_index):
        super().__init__(records, {**config, "train_series_mirror": None}, training=True)
        self.pass_index = pass_index

    def __getitem__(self, index):
        record = self.records[index]
        images, valid = self._read(record, np.random.default_rng([self.pass_index, index]))
        return {"images": images, "valid": valid, "study_uid": record.study_uid}


def state(run, which):
    blob = torch.load(FT / run / which, map_location="cpu", weights_only=False)
    return blob["trainable_state_dict"]


def average(*states):
    keys = states[0].keys()
    assert all(s.keys() == keys for s in states), "trainable key sets differ"
    return {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jitter-passes", type=int, default=4)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else W / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((FT / A30 / "config.json").read_text())
    for other in (AUG, U9):
        c = json.loads((FT / other / "config.json").read_text())
        for k in ("image_model", "image_model_revision", "image_unfreeze_last", "head_hidden", "pooling", "image_size",
                  "windows_per_plane", "slice_band"):
            assert c[k] == config[k], (other, k)

    groups, targets, _, _ = ft.load_population(config, ft.ROOT)
    test_rows, test_y, test_m, test_identity = load_test_population(
        ft.ROOT / config["cohort_manifest"], ft.ROOT / config["lm_labels_csv"], ft.ROOT / config["gold_labels_csv"])
    sets = {"lm_test": test_rows, "gold": groups["gold"]}

    backbone, _ = ft.load_backbone(config)
    model = ft.WindowModel(backbone, config["head_hidden"], config["dropout"], config["pooling"],
                           window_groups=config.get("window_groups")).to("cuda")

    def score(name, loader_for):
        if (out / f"{name}.npz").exists():
            print(name, "exists, skipped", flush=True); return
        t0, scored = time.perf_counter(), {}
        for split, rows in sets.items():
            probs, uids = ft.predict(model, loader_for(rows), "cuda", config["amp"])
            order = [{u: i for i, u in enumerate(uids)}[r.study_uid] for r in rows]
            scored[split] = probs[order]
        lm = ft.masked_macro_auroc(scored["lm_test"], test_y, test_m)
        gold = ft.score_predictions(targets["gold"], scored["gold"])
        np.savez_compressed(out / f"{name}.npz", lm_test_uids=np.asarray(test_identity["study_uids"]),
                            lm_test=scored["lm_test"], gold_uids=np.asarray([r.study_uid for r in groups["gold"]]),
                            gold=scored["gold"], gold_targets=targets["gold"])
        line = {"variant": name, "lm_test": lm["macro_auroc"], "gold": gold["macro_auroc"],
                "seconds": round(time.perf_counter() - t0)}
        with (out / "summary.jsonl").open("a") as h:
            h.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)

    centre = lambda rows: ft._loader(rows, config, training=False)
    a30_best = state(A30, "best.pt")
    variants = {
        "a30-best": a30_best,  # sanity: must reproduce A30's predictions.npz
        "a30-last": state(A30, "last.pt"),
        "a30-avg-best-last": average(a30_best, state(A30, "last.pt")),
        "soup-a30-a30aug": average(a30_best, state(AUG, "best.pt")),
        "soup-a30-u9": average(a30_best, state(U9, "best.pt")),
    }
    for name, sd in variants.items():
        missing = model.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        score(name, centre)

    model.load_state_dict(a30_best, strict=False)
    for p in range(args.jitter_passes):
        def jitter(rows, p=p):
            return torch.utils.data.DataLoader(JitterDataset(rows, config, p), batch_size=1, shuffle=False,
                                               num_workers=config["num_workers"],
                                               collate_fn=lambda b: ((b[0]["images"][None], b[0]["valid"][None]),
                                                                     [b[0]["study_uid"]]))
        score(f"a30-jitter-{p}", jitter)


if __name__ == "__main__":
    main()
