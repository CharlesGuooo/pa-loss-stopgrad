#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Freeze Phase 0.10 severe-event benchmark manifests and splits."""

import hashlib
import json
import random
from pathlib import Path


def freeze_severe_benchmark(
    severe_event_json="outputs/phase0_9/severe_events.json",
    stratification_summary_json="outputs/phase0_9/stratification_summary.json",
    output_dir="outputs/phase0_10",
    val_fraction=0.2,
    split_seed=17,
):
    severe_path = Path(severe_event_json)
    events = json.loads(severe_path.read_text(encoding="utf-8"))
    stratification_summary = _load_optional_json(stratification_summary_json)
    train_events, val_events = split_events(events, val_fraction=val_fraction, seed=split_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_events_json": output_path / "severe_train_events.json",
        "val_events_json": output_path / "severe_val_events.json",
        "manifest_json": output_path / "severe_benchmark_manifest.json",
    }
    paths["train_events_json"].write_text(json.dumps(train_events, indent=2), encoding="utf-8")
    paths["val_events_json"].write_text(json.dumps(val_events, indent=2), encoding="utf-8")

    manifest = {
        "source_event_json": str(severe_path),
        "stratification_summary_json": str(Path(stratification_summary_json)),
        "event_hash": hash_events(events),
        "num_events": len(events),
        "num_train_events": len(train_events),
        "num_val_events": len(val_events),
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "thresholds": stratification_summary.get("thresholds", {}),
        "train_events_json": str(paths["train_events_json"]),
        "val_events_json": str(paths["val_events_json"]),
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "train_events": train_events, "val_events": val_events, "paths": paths}


def split_events(events, val_fraction=0.2, seed=17):
    events = [dict(event) for event in events]
    indices = list(range(len(events)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(events) * val_fraction))) if len(events) > 1 else 0
    val_indices = set(indices[:val_count])
    train_events = [event for index, event in enumerate(events) if index not in val_indices]
    val_events = [event for index, event in enumerate(events) if index in val_indices]
    return train_events, val_events


def hash_events(events):
    canonical = json.dumps(events, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
