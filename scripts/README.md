# Scripts

Run every script from the repository root, for example
`.venv/bin/python scripts/training/train_contrastive.py --help`. Relative paths in
matrices and configurations resolve from there. Every Python script answers `--help` with
its docstring (what it reads, what it writes, how to run it) without doing any work.

Each script is one command. Scripts never import each other. Code shared by more
than one script lives under `src/`: `src/contrastive_learning/` holds the pipeline,
`src/experiment_tools/` the analysis helpers, and `src/digest_site/` the site
builder. A script reaches `src/` with one line:
`sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))`.

## `data/`: cohorts, caches, translations

| Script | What it does |
| --- | --- |
| `build_contrastive_manifest.py` | Freeze a cohort: pair reports with DICOM series, resolve the text policy, assign study-grouped train/validation/test splits, and write the manifest with checksums. Run once per cohort version. |
| `build_window_cache.py` | Pre-build the evaluation window cache of a windowed fine-tuning rung. |
| `migrate_cache_keys.py` | Hardlink cache entries from absolute-path keys to `<study>/<series>` keys (ADR 0001). Prefer rebuilding a cache older than commit `084d086`. |
| `translate_reports_claude.py` | Prepare, run, and merge report translations used by the `translated_english` cohorts. Only `run --execute` calls out to Claude; everything else is offline. |

## `training/`: runs, rungs, heads

| Script | What it does |
| --- | --- |
| `train_contrastive.py` | Train one contrastive run. The single training entry point; every flag is listed by `--help`. |
| `run_measured_training.py` | Wrap `train_contrastive.py` unchanged and also record elapsed time and peak CUDA memory to a telemetry file. Requires CUDA. |
| `run_contrastive_matrix.py` | Run a frozen experiment matrix, or (`--dry-run`) print the exact command for each experiment without executing it. |
| `finetune_supervised.py` | Train one supervised fine-tuning rung from `configs/finetune/<rung>.json`. |
| `run_clip_then_finetune.py` | The OmniRad CLIP pre-training and its matched supervised follow-up, as one command. |
| `train_lm_head.py` | Fit an LM-label head on cached frozen-tower features (`configs/kaggle/lm-*.json`). |
| `train_setting_heads.py` | Fit heads on per-setting features (selected planes, their fat-suppression counterparts, slot mean, per-label slot attention) over several seeds, after checking that the rung's own head reproduces its saved predictions. |
| `profile_lora_memory.py` | Profile GPU memory of LoRA fine-tuning across ranks and layer depths. |
| `run_remaining_declip.sh` | The 2026-09 DeCLIP chain runner, kept for its exact command sequence. |

## `evaluation/`: scoring and features

| Script | What it does |
| --- | --- |
| `evaluate_contrastive.py` | Score one saved checkpoint on a split; write its metrics report and per-query rankings. |
| `rescore_contrastive_checkpoints.py` | Re-evaluate every saved `best.pt` under current metric definitions and summarize which checkpoints moved. |
| `finalize_contrastive_matrix.py` | Verify every run's best epoch against its own history, write the immutable `validation-selection.json`, and optionally spend the locked test split. |
| `evaluate_downstream_probe.py` | The gold-58 downstream probe: cross-validated per-label logistic probes, pooled out-of-fold AUROC, paired bootstrap intervals. The contract itself is `experiment_tools.gold_probe`. Never loads the locked test split. |
| `eval_lm_proxy_heads.py` | Score exported bundles with LM-verdict AUROC on the locked test split and compare them. |
| `extract_omnirad_probe_features.py` | Frozen K8 features from unadapted OmniRad-base and its CLIP-adapted image tower. |
| `extract_setting_features.py` | Per-window features for up to six series per study, under a rung's adapted backbone and its pretrained starting point. Fsynced shards; a rerun resumes. |

## `analysis/`: paired comparisons and follow-ups

All read-only over finished runs. The four paired fine-tuning trials (`compare_clip_init_u9`,
`compare_image_augmentation`, `compare_lateral_rebalance`, `compare_slice_band`) share
`experiment_tools.paired_runs`; the gold-probe comparisons use `experiment_tools.gold_probe`;
the tower-size scripts read their four pinned runs from `experiment_tools.tower_size_runs`.

| Script | What it does |
| --- | --- |
| `compare_downstream_arms.py` | Pair two downstream-probe arms against each other instead of against unadapted DINOv2. |
| `compare_finetune_rungs.py` | Pair fine-tuned rungs with R0 on LM-test and gold-58, one study bootstrap per metric. |
| `compare_headonly_vs_finetune.py` | Head-only readout on frozen towers against fully label-fine-tuned run-5 rungs. |
| `compare_clip_init_u9.py` | Paired LM-test and gold-58 deltas for the CLIP-init u9 run, alone and as an ensemble member. |
| `compare_image_augmentation.py` | Paired deltas for the image-augmentation trial against its matched run. |
| `compare_lateral_rebalance.py` | Paired deltas for the lateral/synovitis rebalancing arms against u9-b1. |
| `compare_slice_band.py` | Paired deltas for the wide slice-band trial against its matched run. |
| `compare_tower_size_factorial.py` | The four pinned tower-size arms on identical gold out-of-fold folds. |
| `plot_tower_size_factorial.py` | Training curves of the tower-size arms as `digest/tower-size-factorial-curves.svg`. |
| `render_tower_size_factorial_html.py` | Render `digest/tower-size-factorial-v1.md` to HTML. |
| `analyze_label_similarity.py` | Summarize the label-similarity-v1 arms into one evidence JSON. |
| `analyze_omnirad_probe.py` | Whether plain CLIP helped OmniRad-base: readout and label grouping on frozen features. |
| `analyze_plane_pooling.py` | Why learned plane attention reads labels worse than averaging the same three planes. |
| `analyze_text_screen_followups.py` | Mojibake and retrieval-hub follow-ups to the English/medical text review. |

## `audits/`: data and preprocessing checks

All read-only. Cohort sampling is in `experiment_tools.cohort_audit`; series hashing and
header reads are in `experiment_tools.dicom_series`.

| Script | What it does |
| --- | --- |
| `audit_integrity_coverage.py` | Patient independence, join coverage and duplicate scope (§7). |
| `audit_gold_case_inventory.py` | What the 58 gold studies can and cannot support (§9). |
| `audit_acquisition_duplicates.py` | Header-only near-duplicate screen over all 4,407 studies, test split included. |
| `verify_duplicate_groups.py` | Confirm or refute the screen's candidate duplicate groups by pixels. |
| `decode_gated_duplicate_candidate.py` | Settle the one train/test duplicate candidate the screen left gated. |
| `audit_acquisition_slots.py` | Recover weighting and fat suppression per series, then measure slot fill rates. |
| `audit_fov_and_depth.py` | Check two preprocessing assumptions against the DICOM itself. |
| `audit_clip_window_mirror.py` | Check every training-series path of a CLIP config through the actual K8 window reader; report missing or empty series. |
| `check_protocol_leak.py` | Whether the acquisition protocol leaks the label. |
| `check_slice_grouping.py` | Depth coverage of the current sampler against contiguous tiled groups. |
| `measure_crop_retention.py` | How much tissue a centred physical crop keeps, for candidate crop widths. |

## `digests/`: reports under `digest/`

Each `build_*_digest.py` rewrites its own files in `digest/` from local, gitignored
artifacts and touches nothing else. Shared pieces: `experiment_tools.curves` (metrics
reader, run summary, static chart), `experiment_tools.digest_charts` (interactive chart),
`experiment_tools.digest_arms` (arm-comparison helpers), and the page CSS and JavaScript
in `templates/`.

| Script | What it does |
| --- | --- |
| `build_digest_site.py` | Build the navigable site under `digest/site/`; `export` first refreshes its data snapshots from `checkpoints/`. |
| `build_project_digest.py` | Rebuild `digest/project-state.html` from `templates/project-digest.html`. |
| `build_both_block_digest.py`, `build_declip_digest.py`, `build_codec_declip_digest.py` | Both-block adaptation, DeCLIP alignment, and the codec-declip-v1 replicate. |
| `build_extended_curves.py`, `build_convergence_curves.py` | Curves pages for the image-block variants and their convergence audit. |
| `build_factor_isolation_digest.py`, `build_combined_lora_digest.py` | Factor-isolation and combined CLIP + LoRA + K8 + 336 results, verified through `experiment_tools.lm_heads`. |
| `build_next_model_pairs_digest.py`, `build_lm_head_tower_size_58_digest.py`, `build_size_matched_digest.py` | Tower-swap and tower-size studies. |
| `build_finetune_ladder_digest.py`, `build_ensemble_selection_digest.py`, `build_joint_loss_digest.py`, `build_omnirad_probe_digest.py` | Fine-tuning ladder, ensemble selection, joint-loss run, OmniRad frozen probe. |
| `build_leaderboard_digest.py` | Every recorded Kaggle public-leaderboard submission. |

## `submission/`: Kaggle bundles

| Script | What it does |
| --- | --- |
| `export_finetune_bundle.py` | Export windowed fine-tuning runs as one Kaggle bundle. |
| `assemble_ensemble_bundle.py` | Combine exported single-member bundles into one multi-member bundle. |
| `build_kaggle_bundle.py` | Benchmark trained contrastive encoders, export a self-contained bundle, and print its LM-test macro AUROC. |
| `predict_kaggle.py` | Run a bundle as the Kaggle kernel does. Copied into every bundle. |

## `archive/`

Provenance only. See [archive/README.md](archive/README.md).
