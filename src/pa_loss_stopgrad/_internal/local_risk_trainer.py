#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config-driven local SimplePredictor training for Phase 0.10."""

import json
import math
from pathlib import Path

import torch
import yaml

from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
from pa_loss_stopgrad.pa_loss.planning_loss import PlanningAwareLoss
from pa_loss_stopgrad._internal.real_data_risk_probe import build_real_data_risk_batch, load_event_index
from pa_loss_stopgrad.eval.risk_metrics import collision_proxy_risk, mode_diversity, summarize_epoch, wta_l1_error
from pa_loss_stopgrad.carriers.simple_predictor import SimplePredictor


def run_local_risk_training(
    data_root=None,
    version="v1.0-trainval",
    train_event_json="outputs/phase0_10/severe_train_events.json",
    val_event_json="outputs/phase0_10/severe_val_events.json",
    output_dir="outputs/phase0_10/local_train",
    config_path="configs/phase0_risk_probe.yaml",
    device="auto",
    epochs=None,
    batch_size=None,
    history_len=None,
    future_len=None,
    max_agents=16,
    hidden_dim=None,
    num_modes=None,
    learning_rate=1e-3,
    lambda_plan=None,
    seed=None,
):
    config = _load_config(config_path)
    seed = int(seed if seed is not None else config.get("seed", 7))
    torch.manual_seed(seed)
    device_obj = _resolve_device(device if device is not None else config.get("device", "cpu"))
    batch_size = int(batch_size if batch_size is not None else config.get("data", {}).get("batch_size", 4))
    history_len = int(history_len if history_len is not None else config.get("data", {}).get("history_len", 5))
    future_len = int(future_len if future_len is not None else config.get("data", {}).get("future_len", 6))
    hidden_dim = int(hidden_dim if hidden_dim is not None else config.get("model", {}).get("hidden_dim", 64))
    num_modes = int(
        num_modes if num_modes is not None else config.get("model", {}).get("num_prediction_modes", 4)
    )
    lambda_plan = float(lambda_plan if lambda_plan is not None else config.get("loss", {}).get("lambda_plan", 0.5))
    epochs = int(epochs if epochs is not None else 3)

    nusc = NuScenesMini(data_root=data_root, version=version)
    train_events = load_event_index(train_event_json)
    val_events = load_event_index(val_event_json)
    predictor = SimplePredictor(state_dim=9, hidden_dim=hidden_dim, num_modes=num_modes, horizon=future_len).to(
        device_obj
    )
    loss_fn = PlanningAwareLoss(
        w_collision=float(config.get("loss", {}).get("w_collision", 1.0)),
        w_comfort=float(config.get("loss", {}).get("w_comfort", 0.0)),
        w_offroad=float(config.get("loss", {}).get("w_offroad", 0.0)),
        tau=float(config.get("loss", {}).get("tau", 1.0)),
    )
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)

    output_path = Path(output_dir)
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    train_metrics = []
    val_metrics = []
    best_val_loss = math.inf
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        predictor.train()
        train_metrics.append(
            _run_epoch(
                predictor,
                loss_fn,
                nusc,
                train_events,
                batch_size=batch_size,
                history_len=history_len,
                future_len=future_len,
                max_agents=max_agents,
                device=device_obj,
                optimizer=optimizer,
                lambda_plan=lambda_plan,
                epoch=epoch,
            )
        )
        predictor.eval()
        with torch.no_grad():
            val_metric = _run_epoch(
                predictor,
                loss_fn,
                nusc,
                val_events,
                batch_size=batch_size,
                history_len=history_len,
                future_len=future_len,
                max_agents=max_agents,
                device=device_obj,
                optimizer=None,
                lambda_plan=lambda_plan,
                epoch=epoch,
            )
        val_metrics.append(val_metric)

        checkpoint = checkpoint_path / f"epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "model_state_dict": predictor.state_dict(), "val_loss": val_metric["loss"]}, checkpoint)
        if val_metric["loss"] < best_val_loss:
            best_val_loss = val_metric["loss"]
            best_checkpoint = checkpoint

    result = {
        "device": str(device_obj),
        "epochs": epochs,
        "num_checkpoints": epochs,
        "best_checkpoint_path": best_checkpoint,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "output_dir": output_path,
    }
    serializable = {
        **result,
        "best_checkpoint_path": str(best_checkpoint) if best_checkpoint is not None else None,
        "output_dir": str(output_path),
    }
    (output_path / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return result


def _run_epoch(
    predictor,
    loss_fn,
    nusc,
    events,
    batch_size,
    history_len,
    future_len,
    max_agents,
    device,
    optimizer,
    lambda_plan,
    epoch,
):
    records = []
    total_loaded = 0
    total_skipped = 0
    for step_events in _event_batches(events, batch_size):
        batch = build_real_data_risk_batch(
            nusc,
            step_events,
            history_len=history_len,
            future_len=future_len,
            max_agents=max_agents,
        )
        total_loaded += batch["num_loaded_events"]
        total_skipped += batch["num_skipped_events"]
        if batch["num_loaded_events"] == 0:
            continue
        batch = _move_batch_to_device(batch, device)
        output = predictor(batch["agent_history"])
        valid = _valid_agent_view(batch, output["predicted_futures"], output["prediction_scores"])
        wta_loss = wta_l1_error(valid["predicted_futures"], valid["gt_futures"])
        pa_loss = loss_fn(
            valid["predicted_futures"],
            valid["ego_plans"],
            None,
            prediction_scores=valid["prediction_scores"],
        )["total_loss"]
        loss = wta_loss + lambda_plan * pa_loss
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        records.append(
            {
                "loss": float(loss.detach().cpu().item()),
                "wta_l1": float(wta_loss.detach().cpu().item()),
                "pa_loss": float(pa_loss.detach().cpu().item()),
                "collision_risk": float(
                    collision_proxy_risk(valid["predicted_futures"].detach(), valid["ego_plans"]).cpu().item()
                ),
                "mode_diversity": float(mode_diversity(valid["predicted_futures"].detach()).cpu().item()),
            }
        )

    return {
        "epoch": epoch,
        "num_loaded_events": total_loaded,
        "num_skipped_events": total_skipped,
        **_zero_metrics_if_empty(summarize_epoch(records)),
    }


def _valid_agent_view(batch, predicted_futures, prediction_scores):
    valid_indices = batch["agent_mask"].nonzero(as_tuple=False)
    batch_indices = valid_indices[:, 0]
    agent_indices = valid_indices[:, 1]
    return {
        "predicted_futures": predicted_futures[batch_indices, agent_indices].unsqueeze(1),
        "prediction_scores": prediction_scores[batch_indices, agent_indices].unsqueeze(1),
        "gt_futures": batch["gt_futures"][batch_indices, agent_indices].unsqueeze(1),
        "ego_plans": {
            "trajectories": batch["ego_plans"]["trajectories"][batch_indices],
            "scores": batch["ego_plans"]["scores"][batch_indices],
        },
    }


def _event_batches(events, batch_size):
    return [events[index : index + batch_size] for index in range(0, len(events), batch_size)]


def _move_batch_to_device(batch, device):
    moved = dict(batch)
    for key in ("agent_history", "gt_futures", "agent_mask", "is_high_conflict"):
        moved[key] = batch[key].to(device)
    moved["ego_plans"] = {key: value.to(device) for key, value in batch["ego_plans"].items()}
    return moved


def _zero_metrics_if_empty(summary):
    defaults = {
        "loss": 0.0,
        "wta_l1": 0.0,
        "pa_loss": 0.0,
        "collision_risk": 0.0,
        "mode_diversity": 0.0,
    }
    return {**defaults, **summary}


def _resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
