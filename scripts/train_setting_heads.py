"""Do the non-fat-suppressed series add label information? Heads on frozen setting features.

Reads the shards of ``extract_setting_features.py`` and, per encoder and seed, fits
four heads on the same studies, targets and selection rule as the rung they came from
(soft LM targets, epoch chosen on validation masked BCE against hard verdicts):

* ``A_primary_mean``: mean of the three primary planes -- what the rung itself reads.
* ``B_concat_secondary``: A's vector, the mean of the present secondary planes (zeros
  if none) and three presence bits, concatenated. The main question.
* ``C_slot_mean``: each present slot's vector plus a learned slot embedding, averaged.
* ``D_label_attention``: 12 label queries attend over the present slot vectors, as
  R4b does over windows. R4b and the contrastive plane attention both lost to a mean,
  so this arm is expected to lose too.

A sanity gate runs first on the finetuned encoder: the rung's own head over arm A's
features must reproduce the rung's saved LM-test and gold predictions. If it does not,
the extraction differs from the rung's evaluation path and every arm is suspect.

Output: ``--output`` JSON with per-arm, per-seed scores, paired deltas against arm A
with study-bootstrap intervals, and per-label deltas; training histories alongside.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contrastive_learning import finetune
from contrastive_learning.artifacts import read_json, write_json
from contrastive_learning.lm_proxy import load_test_population, masked_macro_auroc
from contrastive_learning.lm_supervision import load_population
from contrastive_learning.submission import LABELS, score_predictions
from contrastive_learning.supervised_head import masked_bce

ARMS = ("A_primary_mean", "B_concat_secondary", "C_slot_mean", "D_label_attention")
PLANES = 3


def load_shards(directory: Path, encoder: str) -> dict:
    uids, tokens, valid = [], [], []
    for path in sorted(directory.glob("shard-*.npz")):
        with np.load(path) as shard:
            uids.extend(shard["study_uids"].tolist())
            tokens.append(shard[f"tokens_{encoder}"])
            valid.append(shard["valid"])
    tokens, valid = np.concatenate(tokens).astype(np.float32), np.concatenate(valid)
    # Window mean per slot: [S, 6, D], and which slots have any valid window.
    weights = valid.astype(np.float32)
    slots = (tokens * weights[..., None]).sum(2) / np.maximum(weights.sum(-1), 1)[..., None]
    return {"index": {u: i for i, u in enumerate(uids)}, "slots": slots, "present": valid.any(-1)}


def plane_mean(slots, present):
    """``submission.pool_windows``' plane step: mean over present planes, zeros if none."""
    weights = present.astype(np.float32)
    return (slots * weights[..., None]).sum(1) / np.maximum(weights.sum(1), 1)[:, None]


def arm_inputs(features, uids):
    rows = [features["index"][u] for u in uids]
    slots, present = features["slots"][rows], features["present"][rows]
    primary = plane_mean(slots[:, :PLANES], present[:, :PLANES])
    secondary = plane_mean(slots[:, PLANES:], present[:, PLANES:])
    return {
        "A_primary_mean": primary,
        "B_concat_secondary": np.concatenate([primary, secondary, present[:, PLANES:].astype(np.float32)], 1),
        "slots": slots,
        "present": present,
    }


class SlotHead(nn.Module):
    """Arms C and D: slot vectors plus a learned slot embedding, mean- or attention-pooled."""

    def __init__(self, width, slots, pooling, hidden_dims, dropout, key_dim=128):
        super().__init__()
        self.pooling = pooling
        self.slot_embedding = nn.Parameter(torch.zeros(slots, width))
        if pooling == "mean":
            self.head = finetune.dense_head(width, hidden_dims, dropout)
        else:
            self.key = nn.Linear(width, key_dim)
            self.queries = nn.Parameter(torch.randn(len(LABELS), key_dim) / math.sqrt(key_dim))
            self.dropout = nn.Dropout(dropout)
            self.label_weight = nn.Parameter(torch.randn(len(LABELS), width) / math.sqrt(width))
            self.label_bias = nn.Parameter(torch.zeros(len(LABELS)))

    def forward(self, slots, present):
        tokens = slots + self.slot_embedding
        if self.pooling == "mean":
            weights = present.float()
            pooled = (tokens * weights[..., None]).sum(1) / weights.sum(1).clamp(min=1)[:, None]
            return self.head(pooled)
        scores = torch.einsum("lk,bnk->bln", self.queries, self.key(tokens)) / math.sqrt(self.queries.shape[1])
        attention = scores.masked_fill(~present[:, None], float("-inf")).softmax(-1).nan_to_num(0.0)
        pooled = self.dropout(torch.einsum("bln,bnd->bld", attention, tokens))
        return (pooled * self.label_weight).sum(-1) + self.label_bias


def standardiser(train_inputs, arm):
    """Train-split mean and std; slot arms share one over all present slots."""
    if arm in ("C_slot_mean", "D_label_attention"):
        values = train_inputs["slots"][train_inputs["present"]]
    else:
        values = train_inputs[arm]
    mean, std = values.mean(0), values.std(0) + 1e-6
    if arm == "B_concat_secondary":  # leave the presence bits as they are
        mean[-PLANES:], std[-PLANES:] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def tensors(inputs, arm, scale, device):
    mean, std = scale
    if arm in ("C_slot_mean", "D_label_attention"):
        slots = np.where(inputs["present"][..., None], (inputs["slots"] - mean) / std, 0.0)
        return (torch.as_tensor(slots, dtype=torch.float32, device=device),
                torch.as_tensor(inputs["present"], device=device))
    return (torch.as_tensor((inputs[arm] - mean) / std, dtype=torch.float32, device=device),)


def build(arm, width, config):
    hidden, dropout = config["head_hidden"], config["dropout"]
    if arm == "C_slot_mean":
        return SlotHead(width, 2 * PLANES, "mean", hidden, dropout)
    if arm == "D_label_attention":
        return SlotHead(width, 2 * PLANES, "attention", hidden, dropout)
    return finetune.dense_head(width, hidden, dropout)


def predict(model, inputs, batch=512):
    model.eval()
    with torch.inference_mode():
        count = inputs[0].shape[0]
        return torch.cat([torch.sigmoid(model(*(x[i : i + batch] for x in inputs)))
                          for i in range(0, count, batch)]).cpu().numpy()


def fit(arm, data, config, seed, args):
    torch.manual_seed(seed)
    scale = standardiser(data["train"]["inputs"], arm)
    inputs = {split: tensors(d["inputs"], arm, scale, args.device) for split, d in data.items()}
    width = inputs["train"][0].shape[-1]
    model = build(arm, width, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config["weight_decay"])
    y = torch.as_tensor(data["train"]["y"], device=args.device)
    m = torch.as_tensor(data["train"]["m"], device=args.device)
    rates = m.mean(0)
    generator = torch.Generator().manual_seed(seed)
    best, stale, history = {"bce": math.inf}, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator).to(args.device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            loss = masked_bce(model(*(x[index] for x in inputs["train"])), y[index], m[index], rates)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities = predict(model, inputs["validation"])
        logits = torch.logit(torch.as_tensor(probabilities).clamp(1e-6, 1 - 1e-6))
        bce = float(masked_bce(logits, torch.as_tensor(data["validation"]["y"]),
                               torch.as_tensor(data["validation"]["m"])))
        history.append({"epoch": epoch, "train_masked_bce": float(np.mean(losses)), "validation_masked_bce": bce})
        if bce < best["bce"] - 1e-5:
            best = {"bce": bce, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best["state"])
    return {"epoch": best["epoch"], "validation_masked_bce": best["bce"], "history": history,
            "lm_test": predict(model, inputs["lm_test"]), "gold": predict(model, inputs["gold"])}


def sanity_gate(features, data, checkpoint: Path, device):
    """The rung's own head on arm A's finetuned features against its saved predictions."""
    state = torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    head = finetune.dense_head(features["slots"].shape[-1], config["head_hidden"], config["dropout"])
    head.load_state_dict({k[len("head."):]: v for k, v in state["trainable_state_dict"].items()
                          if k.startswith("head.")})
    head = head.to(device)
    saved = np.load(checkpoint / "predictions.npz")
    result = {}
    for split in ("lm_test", "gold"):
        x = torch.as_tensor(data[split]["inputs"]["A_primary_mean"], device=device)
        ours = predict(head, (x,))
        index = {u: i for i, u in enumerate(saved[f"{split}_uids"].tolist())}
        theirs = saved[split][[index[u] for u in data[split]["uids"]]]
        result[split] = {"max_abs_probability_difference": float(np.abs(ours - theirs).max()),
                         "mean_abs_probability_difference": float(np.abs(ours - theirs).mean())}
    return result


def score(probabilities, data):
    lm = masked_macro_auroc(probabilities["lm_test"], data["lm_test"]["y"], data["lm_test"]["m"])
    gold = score_predictions(data["gold"]["y"], probabilities["gold"])
    return {"lm_test": lm["macro_auroc"], "gold": gold["macro_auroc"],
            "lm_test_per_label": lm["per_label"],
            "gold_per_label": {k: v["auroc"] for k, v in gold["per_label"].items()}}


def paired_bootstrap(reference, candidate, data, split, resamples, seed=2030):
    """95% interval of the seed-averaged macro AUROC difference, resampling studies."""
    rng = np.random.default_rng(seed)
    y, m = data[split]["y"], data[split]["m"]
    deltas = []
    for _ in range(resamples):
        index = rng.integers(0, len(y), len(y))
        if split == "gold":
            if (y[index].min(0) == y[index].max(0)).any():
                continue
            a = score_predictions(y[index], reference[index])["macro_auroc"]
            b = score_predictions(y[index], candidate[index])["macro_auroc"]
        else:
            a = masked_macro_auroc(reference[index], y[index], m[index])["macro_auroc"]
            b = masked_macro_auroc(candidate[index], y[index], m[index])["macro_auroc"]
        deltas.append(b - a)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "share_above_zero": float(np.mean(np.asarray(deltas) > 0))}


def paired_vs_reference(result, predictions, data, resamples, reference="A_primary_mean"):
    """Per-seed, seed-averaged and per-label differences of every arm against ``reference``."""
    base = result["arms"][reference]["runs"]
    for arm, entry in result["arms"].items():
        if arm == reference:
            continue
        entry["paired_vs_A"] = {}
        for split in ("lm_test", "gold"):
            per_seed = [run[split] - ref[split] for run, ref in zip(entry["runs"], base)]
            averaged = {name: np.mean([p[split] for p in predictions[name]], 0) for name in (reference, arm)}
            per_label = {label: float(np.mean([
                label_delta(run[f"{split}_per_label"][label], ref[f"{split}_per_label"][label])
                for run, ref in zip(entry["runs"], base)])) for label in LABELS}
            entry["paired_vs_A"][split] = {
                "per_seed_delta": per_seed,
                "mean_delta": float(np.mean(per_seed)),
                "seed_averaged_interval": paired_bootstrap(averaged[reference], averaged[arm], data, split,
                                                           resamples),
                "per_label_delta": per_label,
            }


def label_delta(candidate, reference):
    """A per-label AUROC difference; NaN when either side had a single LM class."""
    return np.nan if candidate is None or reference is None else candidate - reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--encoders", nargs="+", default=["finetuned", "pretrained"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-tolerance", type=float, default=0.02,
                        help="Max probability difference fp16 windows and autocast may explain")
    parser.add_argument("--skip-gate", action="store_true", help="Synthetic smoke tests only")
    args = parser.parse_args()

    meta = read_json(args.features / "splits.json")
    checkpoint = Path(meta["checkpoint"])
    config = read_json(checkpoint / "config.json")
    groups, targets, masks, _ = load_population(config, finetune.ROOT)
    test_rows, test_y, test_m, _ = load_test_population(finetune.ROOT / config["cohort_manifest"],
                                                        finetune.ROOT / config["lm_labels_csv"],
                                                        finetune.ROOT / config["gold_labels_csv"])
    labelled = {
        "train": (groups["train"], None, masks["train"]),
        "validation": (groups["validation"], targets["validation"], masks["validation"]),
        "lm_test": (test_rows, test_y, test_m),
        "gold": (groups["gold"], targets["gold"], masks["gold"]),
    }
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "checkpoint": str(checkpoint), "encoders": {}}
    for encoder in args.encoders:
        features = load_shards(args.features, encoder)
        data = {}
        for split, (rows, y, m) in labelled.items():
            # The extraction's own split lists, which --limit may have shortened.
            wanted = set(meta["splits"][split])
            missing = wanted - features["index"].keys()
            if missing:
                raise ValueError(f"{split}: {len(missing)} extracted studies missing from the shards")
            keep = [i for i, r in enumerate(rows) if r.study_uid in wanted]
            uids = [rows[i].study_uid for i in keep]
            if split == "train":
                y = finetune.soft_targets(finetune.ROOT / config["lm_labels_csv"], uids, m[keep]) \
                    if config["targets"] == "soft" else targets["train"][keep]
            else:
                y = y[keep]
            data[split] = {"uids": uids, "y": y.astype(np.float32), "m": m[keep].astype(np.float32),
                           "inputs": arm_inputs(features, uids)}
        result = {"studies": {s: len(d["uids"]) for s, d in data.items()},
                  "secondary_present_share": {
                      s: data[s]["inputs"]["present"][:, PLANES:].mean(0).round(4).tolist() for s in data},
                  "arms": {}}
        if encoder == "finetuned":
            result["sanity_gate"] = sanity_gate(features, data, checkpoint, args.device)
            print(json.dumps({"sanity_gate": result["sanity_gate"]}), flush=True)
            worst = max(v["max_abs_probability_difference"] for v in result["sanity_gate"].values())
            if worst > args.gate_tolerance and not args.skip_gate:
                write_json(args.output, report | {"encoders": {encoder: result}})
                raise SystemExit(f"Sanity gate failed: max probability difference {worst:.4f} "
                                 f"> {args.gate_tolerance}; the extraction differs from the rung's evaluation")
        predictions = {}
        for arm in args.arms:
            runs = []
            for seed in args.seeds:
                fitted = fit(arm, data, config, seed, args)
                scored = score(fitted, data)
                runs.append({"seed": seed, "epoch": fitted["epoch"],
                             "validation_masked_bce": fitted["validation_masked_bce"],
                             "history": fitted["history"], **scored})
                predictions.setdefault(arm, []).append({"lm_test": fitted["lm_test"], "gold": fitted["gold"]})
                print(json.dumps({"encoder": encoder, "arm": arm, "seed": seed, "epoch": fitted["epoch"],
                                  "lm_test": round(scored["lm_test"], 4), "gold": round(scored["gold"], 4)}),
                      flush=True)
            result["arms"][arm] = {"runs": runs,
                                   "mean_lm_test": float(np.mean([r["lm_test"] for r in runs])),
                                   "mean_gold": float(np.mean([r["gold"] for r in runs]))}
        if "A_primary_mean" in predictions:
            paired_vs_reference(result, predictions, data, args.resamples)
        report["encoders"][encoder] = result
        write_json(args.output, report)
    summary = {e: {a: {"lm_test": round(v["mean_lm_test"], 4), "gold": round(v["mean_gold"], 4),
                       **({"d_lm_test": round(v["paired_vs_A"]["lm_test"]["mean_delta"], 4),
                           "d_gold": round(v["paired_vs_A"]["gold"]["mean_delta"], 4)} if "paired_vs_A" in v else {})}
                   for a, v in r["arms"].items()} for e, r in report["encoders"].items()}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
