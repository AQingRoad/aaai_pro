#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from new_method.router_model import GainRouter, route_metrics, select_threshold, user_group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-free gain router on a user-group train/valid split.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", choices=("logistic", "mlp"), required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--max-trigger-rate", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = torch.load(args.features, map_location="cpu", weights_only=False)
    features = data["features"].float()
    labels = data["labels"].float()
    train_mask, valid_mask = user_group_split(
        data["user_ids"], valid_fraction=args.valid_fraction, seed=args.split_seed
    )
    if labels[train_mask].sum() == 0 or labels[valid_mask].sum() == 0:
        raise ValueError("Both train and valid partitions must contain positive labels")

    mean = features[train_mask].mean(dim=0)
    std = features[train_mask].std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GainRouter(
        features.shape[1], args.model_type, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    negative_count = int(train_mask.sum().item() - labels[train_mask].sum().item())
    positive_count = int(labels[train_mask].sum().item())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(normalized[train_mask], labels[train_mask]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    valid_x = normalized[valid_mask].to(device)
    valid_y = labels[valid_mask].to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No router checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(valid_x)).cpu()
    baseline_ndcg = data["baseline_ndcg"][valid_mask].float()
    cot_ndcg = data["cot_ndcg"][valid_mask].float()
    threshold_info = select_threshold(
        probabilities,
        cot_ndcg - baseline_ndcg,
        max_trigger_rate=args.max_trigger_rate,
    )
    metrics = route_metrics(
        probabilities,
        threshold=float(threshold_info["threshold"]),
        labels=labels[valid_mask],
        baseline_ndcg=baseline_ndcg,
        cot_ndcg=cot_ndcg,
        baseline_rank=data["baseline_rank"][valid_mask],
        cot_rank=data["cot_rank"][valid_mask],
    )
    metrics.update(
        {
            "split": "router_valid",
            "model_type": args.model_type,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_samples": int(train_mask.sum().item()),
            "valid_samples": int(valid_mask.sum().item()),
            "train_positive": positive_count,
            "valid_positive": int(labels[valid_mask].sum().item()),
            "pos_weight": negative_count / positive_count,
            "best_epoch": best_epoch,
            "best_valid_bce": best_loss,
            "max_trigger_rate": args.max_trigger_rate,
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": data["feature_names"],
            "input_dim": features.shape[1],
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "threshold": float(threshold_info["threshold"]),
            "seed": args.seed,
            "split_seed": args.split_seed,
            "valid_metrics": metrics,
            "parameters": vars(args),
        },
        output,
    )
    output.with_suffix(output.suffix + ".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
