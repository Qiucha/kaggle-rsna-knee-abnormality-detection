#!/usr/bin/env python3
"""Read-only monitoring for the run-5 queue; never changes or restarts training."""

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
PIPELINE = "omnirad-clip-run5-v1"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_lines(path):
    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def rows(path):
    result = []
    for line in read_lines(path):
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # An epoch append may be in progress.
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stamp(seconds):
    return datetime.fromtimestamp(seconds, TZ).strftime("%Y-%m-%d %H:%M %Z")


def output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL,
                                       timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def snapshot(root, directory):
    now = time.time()
    queue_dir = root / "checkpoints/finetune"
    pipeline_dir = root / "checkpoints/pipelines" / PIPELINE
    previous = read_json(directory / "status.json")
    names = [line.split("|")[0].strip() for line in read_lines(queue_dir / "queue-2026-10-03.txt")
             if line.strip() and not line.lstrip().startswith("#")]
    fallback_names = {arm.strip() for line in read_lines(queue_dir / "queue-2026-10-03.txt")
                      if line.strip() and not line.lstrip().startswith("#") for arm in line.split("|")}
    completed = {}
    alerts = []
    for line in read_lines(queue_dir / "queue-2026-10-03.out"):
        match = re.match(r"(\S+) rc=(\S+)(?: ran=(\S+))?", line)
        if match:
            name, code, actual = match.groups()
            report = read_json(queue_dir / (actual or name) / "report.json")
            completed[name] = {"return_code": code, "actual_run": actual or name,
                               "lm_test": report.get("lm_test", {}).get("macro_auroc"),
                               "gold": report.get("gold", {}).get("macro_auroc"),
                               "best_epoch": report.get("best_epoch")}
            if code != "0":
                alerts.append(f"Queue entry {name} returned {code}; inspect its log.")

    process_lines = output(["ps", "-eo", "pid=,etimes=,args="]).splitlines()
    queue_alive = any("/bin/bash " in line and "chain-2026-10-03-run5-queue.sh" in line
                      for line in process_lines)
    active = []
    for line in process_lines:
        if "python " not in line or "scripts/finetune_supervised.py --config " not in line:
            continue
        parts = line.strip().split(None, 2)
        command = shlex.split(parts[2])
        config_path = root / command[command.index("--config") + 1]
        config = read_json(config_path)
        name = config.get("name", config_path.stem)
        if name in fallback_names:
            active.append((int(parts[1]), int(parts[0]), name, config))
    # Worker processes inherit argv; the longest-running process is the parent.
    current = max(active, default=None)
    status = read_json(pipeline_dir / "status.json")
    stage = status.get("stage", "unknown")
    gpu = output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                  "--format=csv,noheader,nounits"])
    free_gb = shutil.disk_usage(root).free / 2**30
    if free_gb < 25:
        alerts.append(f"Project filesystem has {free_gb:.1f} GiB free (queue requires 25 GiB).")
    if stage == "failed":
        alerts.append(f"CLIP pipeline failed during {status.get('failed_stage')}: {status.get('error')}")

    progress = None
    measured_finish = None
    observations = previous.get("contrastive_epoch_observations", {})
    if current:
        elapsed, pid, name, config = current
        path = queue_dir / name / "history.jsonl"
        max_epochs = config.get("epochs", 20)
        if config.get("pipeline_script") and stage == "contrastive_pretraining":
            pre = read_json(root / config["contrastive_pretrain_config"])
            path = root / pre["output_dir"] / "metrics.jsonl"
            max_epochs = pre["epochs"]
        history = rows(path)
        epoch = history[-1].get("epoch", 0) if history else 0
        seconds = [row["seconds"] for row in history if row.get("seconds")]
        if seconds:
            epoch_seconds = sum(seconds) / len(seconds)
            measured_finish = path.stat().st_mtime + (max_epochs - epoch) * epoch_seconds + 120
        elif history and stage == "contrastive_pretraining":
            observations.setdefault(str(epoch), path.stat().st_mtime)
            times = sorted((int(key), value) for key, value in observations.items())
            if len(times) > 1:
                first, last = times[0], times[-1]
                epoch_seconds = (last[1] - first[1]) / (last[0] - first[0])
            else:
                started = datetime.strptime(status["updated_at"], "%Y-%m-%d %H:%M:%S %z").timestamp()
                epoch_seconds = (times[-1][1] - started) / epoch
            measured_finish = times[-1][1] + (max_epochs - epoch) * epoch_seconds
        elif not config.get("pipeline_script"):
            measured_finish = now - elapsed + 3.7 * 3600
        elif stage == "supervised_finetuning":
            started = datetime.strptime(status["updated_at"], "%Y-%m-%d %H:%M:%S %z").timestamp()
            measured_finish = started + 3.7 * 3600
        progress = {"name": name, "pid": pid, "elapsed_seconds": elapsed,
                    "stage": stage if config.get("pipeline_script") else "supervised_finetuning",
                    "epochs_completed": epoch, "max_epochs": max_epochs,
                    "latest_epoch_metrics": history[-1] if history else None,
                    "history_file": str(path),
                    "training_progress": read_json(queue_dir / name / "progress.json"),
                    "full_epoch_finish_estimate": stamp(measured_finish) if measured_finish else None}

    finished = (queue_dir / "chain-2026-10-03-run5-queue.done").exists()
    missing_since = previous.get("queue_missing_since") if not queue_alive else None
    if not queue_alive and not finished and len(completed) < len(names):
        missing_since = missing_since or now
        if now - missing_since > 120:
            alerts.append("Queue process disappeared before all queued entries completed.")

    # A long epoch alone is not a stall. Require low GPU activity as well.
    idle_since = None
    if current and gpu:
        try:
            utilization = float(gpu.split(",")[0])
            if utilization < 5:
                idle_since = previous.get("gpu_idle_since") or now
                if now - idle_since > 15 * 60:
                    alerts.append("GPU utilization remained below 5% for 15 minutes while a job is active.")
        except ValueError:
            pass

    state = {"updated_at": stamp(now), "monitor_pid": os.getpid(),
             "queue_alive": queue_alive, "queue_finished": finished,
             "completed": completed, "current": progress,
             "remaining": [name for name in names if name not in completed],
             "contrastive_stage": stage, "gpu_utilization_memory_mib": gpu,
             "free_project_disk_gib": round(free_gb, 1), "alerts": alerts,
             "queue_missing_since": missing_since, "gpu_idle_since": idle_since,
             "contrastive_epoch_observations": observations}
    if stage in ("completed", "completed_pending_seed"):
        state["contrastive_passed"] = status.get("passed")
        state["seed_comparison_pending"] = stage == "completed_pending_seed"
        state["completion_report"] = str(pipeline_dir / "completion-report.md")
    if current and stage == "queued" and not current[3].get("pipeline_script") and "seed2027" in current[2] and measured_finish:
        state["contrastive_start_estimate"] = stamp(measured_finish)
        state["confirmation_finish_planning_window"] = [stamp(measured_finish + hours * 3600)
                                                       for hours in (15.7, 23.7)]
    elif current and stage == "contrastive_pretraining":
        if measured_finish:
            state["confirmation_finish_estimate"] = stamp(measured_finish + 3.7 * 3600)
        profile = read_json(pipeline_dir / "preflight.json")
        state["preflight_compute_hours_excluding_io_validation_checkpoints"] = profile.get("clip", {}).get(
            "projected_pretrain_compute_hours")
    if current and current[3].get("pipeline_script") and stage in ("validation", "preflight", "contrastive_pretraining"):
        started = datetime.strptime(status["started_at"], "%Y-%m-%d %H:%M:%S %z").timestamp()
        state["confirmation_finish_planning_window"] = [stamp(started + hours * 3600) for hours in (15.7, 23.7)]

    change = {"completed": completed, "run": progress["name"] if progress else None,
              "epoch": progress["epochs_completed"] if progress else None,
              "stage": stage, "alerts": alerts, "finished": finished}
    if change != previous.get("last_event"):
        with (directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": stamp(now), **change}) + "\n")
    state["last_event"] = change
    write_json(directory / "status.json", state)
    lines = ["# Run-5 queue progress", "", f"Updated: {state['updated_at']} (every 60 seconds)", "",
             f"Queue running: {queue_alive}; queue finished: {finished}.", "",
             f"GPU: utilization %, used MiB, total MiB: {gpu or 'unavailable'}.", ""]
    if progress:
        lines += [f"Current: **{progress['name']}** ({progress['stage']}).",
                  f"Completed epochs: {progress['epochs_completed']}/{progress['max_epochs']}.",
                  f"Full-epoch finish estimate: {progress['full_epoch_finish_estimate'] or 'pending timing'}.", ""]
        batch_progress = progress.get("training_progress")
        if batch_progress:
            lines += [f"Current epoch {batch_progress['epoch']}: training batch "
                      f"{batch_progress['training_batch']}/{batch_progress['training_batches']}.", ""]
    for key in ("contrastive_start_estimate", "confirmation_finish_planning_window", "confirmation_finish_estimate"):
        if key in state:
            lines += [f"{key.replace('_', ' ')}: {state[key]}", ""]
    lines += ["ETAs assume the full epoch budget; early stopping can finish supervised runs sooner.",
              "The confirmation planning window is provisional until GPU/epoch timing is available.", "",
              "| Completed run | Exit | Best epoch | LM-test AUROC | Gold AUROC |",
              "|---|---|---|---|---|"]
    for name, report in completed.items():
        lines.append(f"| {name} | {report['return_code']} | {report['best_epoch']} | {report['lm_test']} | {report['gold']} |")
    if alerts:
        lines += ["", "## Alerts", ""] + [f"- {alert}" for alert in alerts]
    if state.get("completion_report"):
        result = "pending postponed seed comparison" if state["seed_comparison_pending"] else str(state["contrastive_passed"])
        lines += ["", f"CLIP confirmation completed. Passed: {result}.",
                  f"Report: {state['completion_report']}"]
    lines += ["", "This monitor records local status and alerts; it does not send chat notifications or change training."]
    temporary = directory / "progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(directory / "progress.md")
    return finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    directory = args.project / "checkpoints/pipelines" / PIPELINE / "monitor"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.once:
            snapshot(args.project, directory)
            return
        while True:
            try:
                if snapshot(args.project, directory):
                    return
            except Exception as error:
                print(f"{stamp(time.time())}: monitor error: {error}", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
