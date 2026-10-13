import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contrastive_learning import finetune
from contrastive_learning.config import TrainConfig
from contrastive_learning.model import _enable_gradient_checkpointing_for_dinov2, freeze_then_unfreeze_last
from contrastive_learning.submission import load_encoder


def tiny_backbone():
    from transformers import Dinov2Config, Dinov2Model

    return Dinov2Model(Dinov2Config(hidden_size=12, num_hidden_layers=8, num_attention_heads=3,
                                  image_size=28, patch_size=14)).eval()


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("windows", [0, 8])
def test_omnirad_checkpoint_keeps_all_learned_blocks_when_supervised_depth_changes(tmp_path, monkeypatch, compact, windows):
    from contrastive_learning import model

    torch.manual_seed(12)
    original = tiny_backbone()
    base_state = copy.deepcopy(original.state_dict())
    freeze_then_unfreeze_last(original, 6)
    with torch.no_grad():
        for parameter in original.parameters():
            if parameter.requires_grad:
                parameter.add_(0.03)
    pre = TrainConfig(image_model="test/omnirad", image_model_revision="weights",
                      image_model_format="timm_dinov2", image_model_base="test/dinov2",
                      image_model_base_revision="architecture", image_unfreeze_last=6,
                      image_size=28, text_unfreeze_last=6, windows_per_plane=windows)

    def restore_base(*args, **kwargs):
        assert kwargs["image_model_format"] == "timm_dinov2"
        assert kwargs["image_model_base_revision"] == "architecture"
        result = tiny_backbone()
        result.load_state_dict(base_state)
        return result

    monkeypatch.setattr(model, "load_image_backbone", restore_base)
    weights = (dict(original.named_parameters()) if compact else original.state_dict())
    if compact:
        weights = {n: p for n, p in weights.items() if p.requires_grad}
    path = tmp_path / "best.pt"
    torch.save({"config": pre.as_dict(), ("trainable_state_dict" if compact else "model_state_dict"):
                {"image_tower.backbone." + n: p.detach() for n, p in weights.items()}}, path)
    restored, _, adaptation = load_encoder(path)
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    fine = {key: getattr(pre, key) for key in ("image_model", "image_model_revision", "image_model_format",
                                            "image_model_base", "image_model_base_revision")}
    fine.update(init_checkpoint=str(path), init_unfreeze_last=3, image_unfreeze_last=3,
                image_size=28, crop_mm=130.0, slices_per_plane=3, slice_band=[0.2, 0.8])
    if windows:
        fine["windows_per_plane"] = windows
    backbone, adaptation = finetune.load_backbone(fine)
    assert adaptation["image_unfreeze_last"] == 3
    for i, block in enumerate(backbone.encoder.layer):
        assert any(p.requires_grad for p in block.parameters()) == (i >= 5)
    # All six adapted blocks keep their learned values, including the three now frozen.
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(backbone.state_dict()[name], tensor, rtol=0, atol=0)
    pixels = torch.rand(1, 3, 28, 28)
    torch.testing.assert_close(backbone.eval()(pixels).last_hidden_state,
                               original(pixels).last_hidden_state, rtol=0, atol=0)


def test_checkpointing_preserves_gradients_after_frozen_blocks():
    torch.manual_seed(7)
    plain = tiny_backbone()
    checkpointed = copy.deepcopy(plain)
    _enable_gradient_checkpointing_for_dinov2(checkpointed)
    pixels = torch.rand(2, 3, 28, 28)
    gradients = []
    for backbone in (plain, checkpointed):
        freeze_then_unfreeze_last(backbone, 6)
        backbone.train()
        backbone(pixels).last_hidden_state.square().mean().backward()
        gradients.append({n: p.grad.clone() for n, p in backbone.named_parameters() if p.requires_grad})
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[1][name], gradients[0][name], rtol=1e-4, atol=1e-6)


def test_depth_reset_rejects_lora_checkpoint(monkeypatch):
    monkeypatch.setattr("contrastive_learning.submission.load_encoder", lambda *args:
                        (tiny_backbone(), SimpleNamespace(windows_per_plane=0, image_kwargs={"image_size":28,"crop_mm":130.0,
                         "slices_per_plane":3,"slice_band":(0.2,0.8)}), {"use_lora":True}))
    with pytest.raises(ValueError, match="without LoRA"):
        finetune.load_backbone({"init_checkpoint":"unused", "init_unfreeze_last":3,
                               "image_size":28,"crop_mm":130.0,"slices_per_plane":3,"slice_band":[0.2,0.8]})


def test_timm_config_requires_both_revisions():
    with pytest.raises(ValueError, match="pinned"):
        TrainConfig(image_model_format="timm_dinov2", image_model_revision="weights").validate_training()


@pytest.mark.parametrize("gold_regresses", [False, True])
@pytest.mark.parametrize("seed_pending", [False, True])
def test_pass_rule_uses_paired_populations_seed_noise_and_gold_guard(tmp_path, monkeypatch, gold_regresses, seed_pending):
    import numpy as np
    from contrastive_learning import clip_pipeline

    rng = np.random.default_rng(32)
    targets = rng.integers(0, 2, (40, 12)).astype(float)
    uids = np.asarray([f"u{i}" for i in range(len(targets))])
    fine = {"comparison_baseline":"baseline", "comparison_seed":"seed", "cohort_manifest":"manifest",
            "allow_pending_seed_comparison": True,
            "lm_labels_csv":"lm", "gold_labels_csv":"gold",
            "pass_rule":{"lm_test_ci_lower_gt":0.0, "gold_delta_gt":-0.024,
                         "gain_gt_absolute_seed_difference":True}}
    for name in ("candidate", "baseline", "seed"):
        if name == "seed" and seed_pending:
            continue
        directory = tmp_path / name
        directory.mkdir()
        import json
        (directory / "report.json").write_text(json.dumps({"best_epoch":4,"seconds":3600}))
        pred = targets if name == "candidate" else 1 - targets
        gold = 1 - targets if gold_regresses and name == "candidate" else targets
        np.savez(directory / "predictions.npz", lm_test_uids=uids, lm_test=pred, gold_uids=uids,
                 gold=gold, gold_targets=targets)
    monkeypatch.setattr(clip_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(clip_pipeline, "load_test_population", lambda *args:
                        ([], targets, np.ones_like(targets), {"study_uids":uids.tolist()}))
    intervals = clip_pipeline.paired_intervals
    monkeypatch.setattr(clip_pipeline, "paired_intervals", lambda *args: intervals(*args, resamples=50))
    result = clip_pipeline.compare_to_run5(fine, tmp_path / "candidate", tmp_path)
    assert result["passed"] is (None if seed_pending else not gold_regresses)
    assert result["seed_comparison_pending"] is seed_pending
    assert result["lm_test"]["paired"]["candidate - run5"]["difference"] == 1.0
    assert result["absolute_lm_test_seed_difference"] == (None if seed_pending else 0.0)
    if seed_pending:
        assert result["pass_checks"]["lm_test_gain_exceeds_absolute_seed_difference"] is None
        assert "PENDING SEED COMPARISON" in (tmp_path / "completion-report.md").read_text()
        # Completing the seed later finalizes the same pass rule without retraining.
        import shutil
        shutil.copytree(tmp_path / "baseline", tmp_path / "seed")
        finalized = clip_pipeline.compare_to_run5(fine, tmp_path / "candidate", tmp_path)
        assert finalized["passed"] is (not gold_regresses)
        assert finalized["seed_comparison_pending"] is False
    assert (tmp_path / "completion-report.md").is_file()


def test_pretraining_failure_never_runs_supervised_stage(tmp_path, monkeypatch):
    from contrastive_learning import clip_pipeline

    fine = {"name":"pipeline", "pipeline_status_dir":"status", "contrastive_pretrain_config":"pretrain"}
    monkeypatch.setattr(clip_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(clip_pipeline, "read_json", lambda *args: {})
    monkeypatch.setattr(clip_pipeline, "validate_pipeline", lambda *args:
                        TrainConfig(output_dir=tmp_path / "contrastive"))
    monkeypatch.setattr(clip_pipeline, "preflight", lambda *args: {})

    def fail_pretraining(*args):
        raise RuntimeError("failed pre-training")

    def no_supervised(*args, **kwargs):
        pytest.fail("Supervised training ran after failed pre-training")

    monkeypatch.setattr(clip_pipeline, "run_training", fail_pretraining)
    monkeypatch.setattr(clip_pipeline, "run_finetune", no_supervised)
    with pytest.raises(RuntimeError, match="failed pre-training"):
        clip_pipeline.run_pipeline(fine, tmp_path / "supervised")
    import json
    status = json.loads((tmp_path / "status/status.json").read_text())
    assert (status["stage"], status["failed_stage"]) == ("failed", "contrastive_pretraining")


def test_reuse_pretraining_requires_all_epochs_matching_recipe_and_checkpoint(tmp_path):
    from contrastive_learning import clip_pipeline
    from contrastive_learning.artifacts import sha256_file, write_json, write_jsonl

    pre = TrainConfig(output_dir=tmp_path, epochs=2)
    write_json(tmp_path / "config.json", pre.as_dict())
    (tmp_path / "best.pt").write_bytes(b"selected checkpoint")
    checksum = sha256_file(tmp_path / "best.pt")
    write_jsonl(tmp_path / "metrics.jsonl", [{"epoch": 1, "mean_r@1": 0.2}])
    with pytest.raises(ValueError, match="incomplete"):
        clip_pipeline.reuse_pretraining(pre, checksum)
    write_jsonl(tmp_path / "metrics.jsonl", [{"epoch": 1, "mean_r@1": 0.2}, {"epoch": 2, "mean_r@1": 0.3}])
    with pytest.raises(ValueError, match="checksum"):
        clip_pipeline.reuse_pretraining(pre, "incorrect")
    result = clip_pipeline.reuse_pretraining(pre, checksum)
    assert result["best_mean_r@1"] == 0.3
    assert result["pretraining_reused"] is True
    changed = TrainConfig(output_dir=tmp_path, epochs=2, seed=2027)
    with pytest.raises(ValueError, match="recipe differs"):
        clip_pipeline.reuse_pretraining(changed, checksum)


def test_the_host_mirror_reaches_both_pipeline_stages(tmp_path, monkeypatch):
    import json
    from contrastive_learning import clip_pipeline

    host = tmp_path / "paths.json"
    host.write_text(json.dumps({"train_series_mirror": str(tmp_path / "mirror")}))
    pre = clip_pipeline.pretraining_config({"output_dir": str(tmp_path / "contrastive")}, host)
    assert pre.train_series_mirror == tmp_path / "mirror"
    stated = clip_pipeline.pretraining_config({"train_series_mirror": {"to": "/stated"}}, host)
    assert stated.train_series_mirror == Path("/stated")

    seen = {}
    fine = {"name": "pipeline", "pipeline_status_dir": "status", "contrastive_pretrain_config": "pretrain",
            "init_checkpoint": "contrastive/best.pt", "reuse_completed_contrastive_checkpoint": True}
    monkeypatch.setattr(clip_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(clip_pipeline, "read_json", lambda *args: {})

    def validate(fine, pre_value, paths_config):
        seen["pre_paths"] = paths_config
        return pre

    monkeypatch.setattr(clip_pipeline, "validate_pipeline", validate)
    monkeypatch.setattr(clip_pipeline, "reuse_pretraining", lambda *args: {})
    monkeypatch.setattr(clip_pipeline, "sha256_file", lambda *args: "sha")
    monkeypatch.setattr(clip_pipeline, "run_finetune", lambda config, *args, **kwargs: seen.update(fine=config) or {})
    monkeypatch.setattr(clip_pipeline, "compare_to_run5", lambda *args: {"seed_comparison_pending": True,
                                                                          "passed": None})
    clip_pipeline.run_pipeline(fine, tmp_path / "supervised", paths_config=host)
    assert seen["pre_paths"] == host
    assert seen["fine"]["train_series_mirror"] == {"to": str(tmp_path / "mirror")}


def test_the_run5_recipe_check_ignores_where_this_host_keeps_its_data():
    from contrastive_learning import clip_pipeline

    baseline = {"name": "run5", "seed": 2026, "train_series_mirror": {"to": "/old-host/mirror"}}
    assert clip_pipeline.baseline_differences({"name": "clip", "seed": 2026}, baseline) == []
    assert clip_pipeline.baseline_differences({"name": "clip", "seed": 2027}, baseline) == ["seed"]


def test_a_two_stage_config_runs_its_pipeline_in_process_with_the_host_paths(tmp_path, monkeypatch):
    import importlib.util
    import json
    from contrastive_learning import clip_pipeline

    script = Path(__file__).resolve().parents[1] / "scripts/finetune_supervised.py"
    spec = importlib.util.spec_from_file_location("finetune_supervised_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"contrastive_pretrain_config": "pretrain"}))
    calls = []
    monkeypatch.setattr(clip_pipeline, "run_pipeline", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(module, "run_finetune", lambda *args, **kwargs: pytest.fail("ran a single stage"))
    monkeypatch.setattr("sys.argv", ["finetune_supervised.py", "--config", str(config), "--output",
                                     str(tmp_path / "out"), "--paths-config", str(tmp_path / "paths.json")])
    module.main()
    (fine, output), options = calls[0]
    assert fine == {"contrastive_pretrain_config": "pretrain"} and output == tmp_path / "out"
    assert options == {"device": "cuda", "paths_config": tmp_path / "paths.json"}


def test_reusing_pretraining_ignores_a_moved_series_mirror(tmp_path):
    from contrastive_learning import clip_pipeline
    from contrastive_learning.artifacts import sha256_file, write_json, write_jsonl

    recorded = TrainConfig(output_dir=tmp_path, epochs=1).as_dict()
    write_json(tmp_path / "config.json", {**recorded, "train_series_mirror": {"to": "/old-host/mirror"}})
    (tmp_path / "best.pt").write_bytes(b"selected checkpoint")
    write_jsonl(tmp_path / "metrics.jsonl", [{"epoch": 1, "mean_r@1": 0.2}])
    pre = TrainConfig(output_dir=tmp_path, epochs=1, train_series_mirror=tmp_path / "mirror")
    assert clip_pipeline.reuse_pretraining(pre, sha256_file(tmp_path / "best.pt"))["pretraining_reused"]
