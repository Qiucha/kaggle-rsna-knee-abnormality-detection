"""Fine-tune the image backbone end to end on LM labels: one ladder rung per config."""

import argparse
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contrastive_learning.artifacts import read_json
from contrastive_learning.finetune import run_finetune, with_host_mirror


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths-config", type=Path,
                        help="Host paths file (default: $CONTRASTIVE_PATHS_CONFIG, then paths.json at the repository root)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new experiment directory")
    config = read_json(args.config)
    if config.get("pipeline_script"):
        # A queued two-stage experiment uses its isolated, tested source tree.
        raise SystemExit(subprocess.call([
            # A relative path is relative to the repository root, not the working directory.
            sys.executable, str(ROOT / config["pipeline_script"]), "--config", str(args.config),
            "--output", str(args.output), "--device", args.device,
        ]))
    config = with_host_mirror(config, args.paths_config)
    run_finetune(config, args.output, device=args.device)


if __name__ == "__main__":
    main()
