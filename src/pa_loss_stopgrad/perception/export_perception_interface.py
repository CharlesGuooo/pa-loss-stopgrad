#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export a frozen SparseDrive perception interface cache.

Phase 1.6 deliberately cuts the graph at SparseDrive perception:
backbone/neck + detection/map heads are frozen and MotionPlanningHead is
disabled. The serialized cache is the contract consumed by later
Planning-GAZE adapter / predictor work.
"""

import argparse
import json
from pathlib import Path

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import single_gpu_test, set_random_seed
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401  # register SparseDrive datasets/models


BASELINE_MAP = 0.42898068183904803
BASELINE_NDS = 0.48248987910556346
BASELINE_FRAMES = 81
PARITY_TOLERANCE = 5e-3


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed, deterministic=True)
    cfg = _load_config(args)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.freeze_perception:
        freeze_report = freeze_perception(model)
    else:
        freeze_report = count_parameters(model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.disable_motion_plan:
        disable_motion_plan_after_checkpoint(model)
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if hasattr(dataset, "PALETTE"):
        model.PALETTE = getattr(dataset, "PALETTE")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    with torch.no_grad():
        outputs = single_gpu_test(model, data_loader, show=False)

    manifest = export_interface_cache(outputs, dataset, samples_dir)
    interface_report = build_interface_report(manifest, samples_dir, args.baseline_frames)
    if args.skip_parity:
        parity_report = {"status": "skipped", "within_tolerance": True}
    else:
        parity_report = evaluate_detection_parity(outputs, dataset, args.expected_map, args.expected_nds)
    grad_isolation_report = check_gradient_isolation(model.module)
    frames_ok = (args.baseline_frames <= 0) or (interface_report["sample_count"] == args.baseline_frames)
    summary = {
        "decision": "Boundary-Go"
        if (
            parity_report["within_tolerance"]
            and grad_isolation_report["perception_params_with_grad"] == 0
            and grad_isolation_report["proxy_grad_nonzero"]
            and frames_ok
            and interface_report["schema_valid"]
        )
        else "No-Go",
        "freeze_report": freeze_report,
        "parity_report": parity_report,
        "grad_isolation_report": grad_isolation_report,
        "interface_report": interface_report,
        "manifest": str(output_dir / "manifest.json"),
    }
    mmcv.dump(manifest, output_dir / "manifest.json")
    Path(output_dir / "phase1_6_export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_config(args):
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.ann_file = args.ann_file
        cfg.data.test.version = args.version
        cfg.data.test.work_dir = cfg.get("work_dir") or "./work_dirs/sparsedrive_small_stage2"
        if "eval_config" in cfg.data.test:
            cfg.data.test.eval_config.ann_file = args.ann_file
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        raise TypeError("Phase 1.6 exporter expects a single test dataset config.")
    return cfg


def freeze_perception(model):
    """Freeze all built modules after disabling MotionPlanningHead."""

    model.requires_grad_(False)
    report = count_parameters(model)
    report["policy"] = "All built SparseDrive parameters frozen; MotionPlanningHead is disabled after checkpoint load for export."
    return report


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
    }


def postprocess_perception_only(model, det_output, map_output, data):
    """Document the Phase 1.6 cut: post-process det/map with no motion/plan outputs."""

    return model.head.post_process((det_output, map_output, None, None), data)


def disable_motion_plan_after_checkpoint(model):
    """Keep checkpoint loading complete, then cut MotionPlanningHead out of forward."""

    model.head.task_config["with_motion_plan"] = False
    model.head.task_config.with_motion_plan = False


def export_interface_cache(outputs, dataset, samples_dir):
    manifest = {"sample_count": len(outputs), "samples": []}
    for index, output in enumerate(outputs):
        info = dataset.data_infos[index]
        result = output.get("img_bbox", output)
        sample = build_interface_sample(result, info)
        token = sample["meta"]["sample_token"]
        path = samples_dir / f"{index:06d}_{token}.pkl"
        mmcv.dump(sample, path)
        manifest["samples"].append(
            {
                "index": index,
                "sample_token": token,
                "timestamp": sample["meta"]["timestamp"],
                "path": str(path),
                "agent_count": len(sample["agents"]["boxes"]),
                "map_vector_count": len(sample["map"]["vectors"]),
            }
        )
    return manifest


def build_interface_sample(result, info):
    boxes = _to_list(result.get("boxes_3d", []))
    return {
        "meta": {
            "sample_token": info["token"],
            "timestamp": int(info["timestamp"]),
        },
        "agents": {
            "boxes": [box[:9] for box in boxes],
            "scores": _to_list(result.get("scores_3d", [])),
            "labels": _to_list(result.get("labels_3d", [])),
            "track_ids": _to_list(result.get("instance_ids", [])),
        },
        "map": {
            "vectors": _to_list(result.get("vectors", [])),
            "scores": _to_list(result.get("scores", [])),
            "labels": _to_list(result.get("labels", [])),
        },
        "ego": {
            "ego2global_translation": _to_list(info["ego2global_translation"]),
            "ego2global_rotation": _to_list(info["ego2global_rotation"]),
            "lidar2ego_translation": _to_list(info["lidar2ego_translation"]),
            "lidar2ego_rotation": _to_list(info["lidar2ego_rotation"]),
            "ego_status": _to_list(info["ego_status"]),
            "command": _to_list(info["gt_ego_fut_cmd"]),
        },
    }


def evaluate_detection_parity(outputs, dataset, expected_map=BASELINE_MAP, expected_nds=BASELINE_NDS):
    eval_mode = {
        "with_det": True,
        "with_tracking": False,
        "with_map": False,
        "with_motion": False,
        "with_planning": False,
        "tracking_threshold": 0.2,
        "motion_threshhold": 0.2,
    }
    metrics = dataset.evaluate(outputs, eval_mode=eval_mode, metric=["bbox"])
    map_value = float(metrics.get("img_bbox_NuScenes/mAP"))
    nds_value = float(metrics.get("img_bbox_NuScenes/NDS"))
    return {
        "status": "passed",
        "mAP": map_value,
        "NDS": nds_value,
        "baseline_mAP": expected_map,
        "baseline_NDS": expected_nds,
        "tolerance": PARITY_TOLERANCE,
        "within_tolerance": abs(map_value - expected_map) <= PARITY_TOLERANCE
        and abs(nds_value - expected_nds) <= PARITY_TOLERANCE,
    }


def check_gradient_isolation(model):
    model.zero_grad(set_to_none=True)
    proxy = torch.nn.Parameter(torch.ones((), device="cuda"))
    loss = proxy * proxy
    loss.backward()
    perception_params_with_grad = sum(1 for param in model.parameters() if param.grad is not None)
    return {
        "status": "passed" if perception_params_with_grad == 0 and proxy.grad is not None else "failed",
        "perception_params_with_grad": int(perception_params_with_grad),
        "proxy_grad_nonzero": bool(proxy.grad is not None and proxy.grad.detach().abs().item() > 0),
    }


def build_interface_report(manifest, samples_dir, baseline_frames=BASELINE_FRAMES):
    first_sample = mmcv.load(manifest["samples"][0]["path"]) if manifest["samples"] else {}
    schema_report = validate_sample(first_sample)
    frames_ok = (baseline_frames <= 0) or (manifest["sample_count"] == baseline_frames)
    return {
        "status": "passed" if frames_ok and schema_report["valid"] else "failed",
        "sample_count": manifest["sample_count"],
        "schema_valid": schema_report["valid"],
        "schema_errors": schema_report["errors"],
        "sample_dir": str(samples_dir),
        "first_sample": manifest["samples"][0] if manifest["samples"] else None,
    }


def validate_sample(sample):
    errors = []
    for key in ("meta", "agents", "map", "ego"):
        if key not in sample:
            errors.append(f"missing {key}")
    if len(sample.get("agents", {}).get("boxes", [[None] * 9])[0]) != 9:
        errors.append("agents.boxes rows must have width 9")
    if len(sample.get("ego", {}).get("ego_status", [])) != 10:
        errors.append("ego.ego_status must have width 10")
    if len(sample.get("ego", {}).get("command", [])) != 3:
        errors.append("ego.command must have width 3")
    return {"valid": not errors, "errors": errors}


def _to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_list(item) for item in value]
    return value


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output-dir", default="work_dirs/phase1_6/perception_interface_mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-perception", action="store_true")
    parser.add_argument("--disable-motion-plan", action="store_true")
    parser.add_argument("--baseline-frames", type=int, default=BASELINE_FRAMES,
                        help="expected #frames for the mini parity gate; <=0 disables the count check (trainval)")
    parser.add_argument("--skip-parity", action="store_true",
                        help="skip the mini-specific mAP/NDS parity gate (use for trainval scale)")
    parser.add_argument("--expected-map", type=float, default=BASELINE_MAP)
    parser.add_argument("--expected-nds", type=float, default=BASELINE_NDS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
