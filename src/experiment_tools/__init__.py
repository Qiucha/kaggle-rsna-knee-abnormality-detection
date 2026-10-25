"""Shared analysis helpers for the scripts under ``scripts/``.

``contrastive_learning`` is the training and inference library. This package holds
what the experiment, audit and digest scripts share with each other, so no script
imports another script.

Probes and comparisons: gold_probe, paired_runs, lm_heads, tower_size_runs.
Audits: cohort_audit, dicom_series.
Digests: curves, digest_charts, digest_arms, joint_loss, text_hubs.
"""
