#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from new_method.router_model import GainRouter, route_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen gain router without tuning its threshold.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = torch.load(args.features, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if data["feature_names"] != checkpoint["feature_names"]:
        raise ValueError("Feature schema differs between training and evaluation")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GainRouter(
        checkpoint["input_dim"],
        checkpoint["model_type"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized = (data["features"].float() - checkpoint["feature_mean"]) / checkpoint[
        "feature_std"
    ]
    with torch.no_grad():
        probabilities = torch.sigmoid(model(normalized.to(device))).cpu()
    metrics = route_metrics(
        probabilities,
        threshold=float(checkpoint["threshold"]),
        labels=data["labels"].float(),
        baseline_ndcg=data["baseline_ndcg"].float(),
        cot_ndcg=data["cot_ndcg"].float(),
        baseline_rank=data["baseline_rank"],
        cot_rank=data["cot_rank"],
    )
    metrics.update(
        {
            "split": data["metadata"]["dataset"],
            "model_type": checkpoint["model_type"],
            "seed": checkpoint["seed"],
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "threshold_source": "router_valid",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
