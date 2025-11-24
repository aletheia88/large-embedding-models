#!/usr/bin/env python3
"""
Train a ResidualMLP restoration model on DINO embeddings for a given corruption scheme/range.

This script wraps the knn_restore utilities to:
1. Extract clean + corrupted embeddings from Imagenette.
2. Train the ResidualMLP predictor.
3. Evaluate reconstruction quality and report k-NN accuracy.
4. Save the trained state_dict to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch

from knn_restore import (
    GlobalConfigs,
    ReconstructConfigs,
    load_dino_encoder,
    get_embeddings_n_labels,
    create_dataloaders,
    train,
    evaluate,
    evaluate_reconstruction,
    knn_predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResidualMLP on DINO features for retrieval imagination.")
    parser.add_argument("--scheme", choices=["baseline", "crop", "occlude", "gaussian"], default="occlude")
    parser.add_argument(
        "--corrupt-range",
        type=float,
        nargs=2,
        required=True,
        metavar=("MIN_FRAC", "MAX_FRAC"),
        help="Area fraction range for corruption when training the MLP.",
    )
    parser.add_argument("--noise-std", type=float, default=None, help="Noise std for gaussian corruption.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--k-neighbors", type=int, default=25, help="k used for k-NN evaluation.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=3, choices=[2, 3])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1912)
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the trained model state_dict.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of metrics; defaults to <output>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    global_configs = GlobalConfigs(
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        scheme=args.scheme,
        corrupt_range=tuple(args.corrupt_range),
        noise_std=args.noise_std,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        generator=generator,
    )
    restore_configs = ReconstructConfigs(
        method="mlp",
        lam=None,
        hidden_dims=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
    )

    print(f"Loading DINO encoder on {device}...")
    encoder_bundle = load_dino_encoder(device)

    print("Extracting clean embeddings...")
    clean = get_embeddings_n_labels(global_configs, encoder_bundle, corruption=False)
    print("Extracting corrupted embeddings...")
    corrupt = get_embeddings_n_labels(global_configs, encoder_bundle, corruption=True)

    train_loader, train_eval_loader, val_loader = create_dataloaders(
        global_configs,
        restore_configs,
        clean["embeddings"]["train"],
        corrupt["embeddings"]["train"],
        clean["embeddings"]["valid"],
        corrupt["embeddings"]["valid"],
    )

    print("Training ResidualMLP...")
    model, losses = train(restore_configs, train_loader, val_loader, device)
    model.eval()
    print("Evaluating embeddings...")
    train_pred, val_pred = evaluate(model, train_eval_loader, val_loader, device)

    train_metrics = evaluate_reconstruction(train_pred, clean["embeddings"]["train"])
    val_metrics = evaluate_reconstruction(val_pred, clean["embeddings"]["valid"])
    print("Train metrics:", train_metrics)
    print("Valid metrics:", val_metrics)

    print("Computing k-NN accuracy...")
    valid_knn_pred = knn_predict(
        train_pred,
        clean["labels"]["train"],
        val_pred,
        k=global_configs.k_neighbors,
        batch_size=global_configs.batch_size,
        device=device,
    )
    accuracy = (valid_knn_pred == clean["labels"]["valid"]).float().mean().item()
    print(f"k-NN accuracy (k={global_configs.k_neighbors}): {accuracy * 100:.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved model weights to {args.output}")

    summary = {
        "scheme": args.scheme,
        "corrupt_range": args.corrupt_range,
        "noise_std": args.noise_std,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "knn_accuracy": accuracy,
        "losses": losses,
        "output": args.output.as_posix(),
    }
    summary_path = args.summary_json or args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
