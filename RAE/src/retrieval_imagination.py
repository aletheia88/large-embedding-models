#!/usr/bin/env python3
"""
Retrieval imagination experiment:
    1. Apply fragment corruption (crop/occlusion/gaussian) to validation images.
    2. Reconstruct clean embeddings with a small ResidualMLP trained on DINO features.
    3. Embed fragments, reconstructions, and clean images with the frozen DINO encoder.
    4. Retrieve top-k nearest clean training images in embedding space.
    5. Report precision@k and dump visualization metadata for Jupyter analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from fastai.vision.all import URLs, untar_data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from corruption import RandomAreaDownUp, RandomSquareOccluder
from knn_restore import load_dino_encoder
from restore_methods import ResidualMLP


def default_data_root(size: str) -> Path:
    url_map = {
        "full": URLs.IMAGENETTE,
        "320px": URLs.IMAGENETTE_320,
        "160px": URLs.IMAGENETTE_160,
    }
    if size not in url_map:
        raise ValueError(f"size must be one of {list(url_map)}, got {size!r}")
    return Path(untar_data(url_map[size]))


class GaussianNoiseTransform:
    """Adds Gaussian noise to a tensor image in [0,1] and clamps the result."""

    def __init__(self, std: float):
        if std <= 0:
            raise ValueError("std must be positive for GaussianNoiseTransform.")
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return (x + noise).clamp(0.0, 1.0)


def build_fragment_transform(
    scheme: str,
    corrupt_range: Optional[Tuple[float, float]],
    gaussian_std: Optional[float],
) -> Optional[object]:
    scheme = scheme.lower()
    if scheme in ("baseline", "none"):
        return None
    if scheme == "crop":
        if corrupt_range is None:
            raise ValueError("crop corruption requires --corrupt-range.")
        return RandomAreaDownUp(out_size=224, area_frac_range=corrupt_range)
    if scheme == "occlude":
        if corrupt_range is None:
            raise ValueError("occlude corruption requires --corrupt-range.")
        return RandomSquareOccluder(area_frac_range=corrupt_range, fill=0.0)
    if scheme == "gaussian":
        if gaussian_std is None:
            raise ValueError("gaussian corruption requires --gaussian-std.")
        return GaussianNoiseTransform(gaussian_std)
    raise ValueError(f"Unknown corruption scheme: {scheme}")


class ImagenetteRetrievalDataset(Dataset):
    """
    Wraps torchvision's ImageFolder to emit clean tensors plus optional fragments.
    Always keeps labels + file paths for downstream visualization.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        base_transform,
        fragment_transform=None,
        return_fragment: bool = False,
    ):
        self.root = Path(root)
        split_root = self.root / split
        self.dataset = datasets.ImageFolder(split_root)
        self.transform = base_transform
        self.fragment_transform = fragment_transform
        self.return_fragment = return_fragment

    def __len__(self) -> int:
        return len(self.dataset.samples)

    @property
    def classes(self) -> List[str]:
        return self.dataset.classes

    def __getitem__(self, idx: int):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path).convert("RGB")
        clean = self.transform(image)
        item = {
            "clean": clean,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }
        if self.return_fragment:
            fragment = clean.clone()
            if self.fragment_transform is not None:
                fragment = self.fragment_transform(fragment)
            item["fragment"] = fragment
        return item


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_residual_mlp(
    ckpt_path: Path,
    hidden_dims: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
):
    model = ResidualMLP(
        in_dim=768,
        hidden_dims=hidden_dims,
        out_dim=768,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def pooled_embeddings(feature_encoder, images: torch.Tensor) -> torch.Tensor:
    model = feature_encoder["model"]
    pixel_values = (images - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)
    outputs = model(pixel_values=pixel_values)
    hidden = outputs.last_hidden_state[:, 1:, :]
    b, n, c = hidden.shape
    hw = int(n**0.5)
    if hw * hw != n:
        raise ValueError(f"Unexpected token count {n}, cannot reshape to square.")
    z = hidden.transpose(1, 2).reshape(b, c, hw, hw)
    z = z.mean(dim=(-2, -1))
    return F.normalize(z, dim=-1)


@torch.no_grad()
def reconstruct_features(mlp: ResidualMLP, feats: torch.Tensor) -> torch.Tensor:
    return F.normalize(mlp(feats), dim=-1)


def stack_batches(chunks: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, device=device)
    cat = torch.cat(chunks, dim=0)
    return cat.to(device)


def build_index(
    feature_encoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    feat_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    paths: List[str] = []
    for batch in tqdm(loader, desc="Indexing train embeddings"):
        clean = batch["clean"].to(device)
        feats = pooled_embeddings(feature_encoder, clean)
        feat_chunks.append(feats)
        label_chunks.append(batch["label"].to(device))
        paths.extend(batch["path"])
    feats = stack_batches(feat_chunks, device)
    labels = stack_batches(label_chunks, device)
    return feats, labels, paths


def cosine_topk(
    query_feats: torch.Tensor,
    index_feats: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sims = query_feats @ index_feats.T
    k = min(k, index_feats.size(0))
    scores, indices = sims.topk(k, dim=-1, largest=True)
    return indices, scores


def precision_at_k(
    retrieved_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> torch.Tensor:
    matches = retrieved_labels == query_labels.unsqueeze(1)
    return matches.float().mean(dim=1)


def tensor_to_png(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0.0, 1.0), path)


def format_retrievals(
    topk_idx: torch.Tensor,
    topk_scores: torch.Tensor,
    train_paths: Sequence[str],
    train_labels_cpu: torch.Tensor,
    class_names: Sequence[str],
) -> List[Dict]:
    rows: List[Dict] = []
    for idx, score in zip(topk_idx.tolist(), topk_scores.tolist()):
        label = int(train_labels_cpu[idx])
        rows.append(
            {
                "path": train_paths[idx],
                "label": label,
                "label_name": class_names[label],
                "score": float(score),
            }
        )
    return rows


def evaluate_queries(
    feature_encoder,
    mlp_model: ResidualMLP,
    loader: DataLoader,
    index_feats: torch.Tensor,
    index_labels: torch.Tensor,
    train_paths: Sequence[str],
    class_names: Sequence[str],
    device: torch.device,
    k: int,
    max_queries: Optional[int],
    num_visualize: int,
    vis_dir: Path,
) -> Tuple[Dict[str, float], List[Dict], int]:
    index_labels_cpu = index_labels.cpu()
    total = 0
    precision_sums = {"fragment": 0.0, "reconstruction": 0.0, "clean": 0.0}
    records: List[Dict] = []

    for batch in tqdm(loader, desc="Evaluating queries"):
        clean = batch["clean"].to(device)
        fragment = batch["fragment"].to(device)
        labels = batch["label"].to(device)

        clean_feats = pooled_embeddings(feature_encoder, clean)
        fragment_feats = pooled_embeddings(feature_encoder, fragment)
        recon_feats = reconstruct_features(mlp_model, fragment_feats)

        eval_pack = {
            "fragment": fragment_feats,
            "reconstruction": recon_feats,
            "clean": clean_feats,
        }
        retrieval_cache = {}
        batch_size = labels.size(0)

        for key, feats in eval_pack.items():
            topk_idx, topk_scores = cosine_topk(feats, index_feats, k)
            retrieval_cache[key] = (topk_idx, topk_scores)
            retrieved_labels = index_labels[topk_idx]
            prec = precision_at_k(retrieved_labels, labels)
            precision_sums[key] += float(prec.sum().item())

        if len(records) < num_visualize:
            for i in range(min(batch_size, num_visualize - len(records))):
                rec = {
                    "query_index": total + i,
                    "label": int(batch["label"][i].item()),
                    "label_name": class_names[int(batch["label"][i].item())],
                    "clean_path": batch["path"][i],
                }
                base_name = f"sample_{total + i:05d}"
                fragment_png = vis_dir / f"{base_name}_fragment.png"
                tensor_to_png(batch["fragment"][i], fragment_png)
                rec["fragment_png"] = fragment_png.as_posix()
                rec["recon_png"] = None
                rec["retrievals"] = {}
                for variant, (top_idx, top_scores) in retrieval_cache.items():
                    rec["retrievals"][variant] = format_retrievals(
                        top_idx[i],
                        top_scores[i],
                        train_paths,
                        index_labels_cpu,
                        class_names,
                    )
                records.append(rec)
                if len(records) >= num_visualize:
                    break

        total += batch_size
        if max_queries is not None and total >= max_queries:
            break

    denom = max(total, 1)
    metrics = {k: precision_sums[k] / denom for k in precision_sums}
    return metrics, records, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval imagination: compare fragment vs reconstruction retrieval."
    )
    parser.add_argument("--device", default=None, help="Torch device (default: auto).")
    parser.add_argument("--data-size", default="320px", choices=["full", "320px", "160px"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scheme", choices=["baseline", "crop", "occlude", "gaussian"], default="occlude")
    parser.add_argument(
        "--corrupt-range",
        type=float,
        nargs=2,
        metavar=("MIN_FRAC", "MAX_FRAC"),
        help="Area fraction for crop/occlude corruptions.",
    )
    parser.add_argument("--gaussian-std", type=float, default=None, help="Std for gaussian corruption.")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval.")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries for quick tests.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrieval_imagination"))
    parser.add_argument("--num-visualize", type=int, default=8, help="How many samples to dump for notebook.")
    parser.add_argument("--mlp-ckpt", type=Path, required=True, help="Path to a trained ResidualMLP checkpoint.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden dimension for the ResidualMLP.")
    parser.add_argument("--mlp-num-layers", type=int, default=3, help="Number of layers in the ResidualMLP.")
    parser.add_argument("--mlp-dropout", type=float, default=0.1, help="Dropout for the ResidualMLP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.results_dir / "viz"
    vis_dir.mkdir(exist_ok=True)

    data_root = default_data_root(args.data_size)
    base_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    fragment_transform = build_fragment_transform(
        args.scheme,
        tuple(args.corrupt_range) if args.corrupt_range else None,
        args.gaussian_std,
    )

    train_dataset = ImagenetteRetrievalDataset(
        data_root, "train", base_transform, return_fragment=False
    )
    val_dataset = ImagenetteRetrievalDataset(
        data_root,
        "val",
        base_transform,
        fragment_transform=fragment_transform,
        return_fragment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    feature_encoder = load_dino_encoder(device)
    mlp_model = load_residual_mlp(
        args.mlp_ckpt,
        args.mlp_hidden_dim,
        args.mlp_num_layers,
        args.mlp_dropout,
        device,
    )
    index_feats, index_labels, train_paths = build_index(feature_encoder, train_loader, device)
    metrics, records, num_processed = evaluate_queries(
        feature_encoder,
        mlp_model,
        val_loader,
        index_feats,
        index_labels,
        train_paths,
        train_dataset.classes,
        device,
        args.k,
        args.max_queries,
        args.num_visualize,
        vis_dir,
    )

    summary = {
        "k": args.k,
        "num_queries": num_processed,
        "scheme": args.scheme,
        "corrupt_range": args.corrupt_range,
        "gaussian_std": args.gaussian_std,
        "precision": metrics,
        "mlp_ckpt": args.mlp_ckpt.as_posix(),
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_num_layers": args.mlp_num_layers,
        "mlp_dropout": args.mlp_dropout,
        "results_dir": args.results_dir.as_posix(),
    }
    metrics_path = args.results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))

    samples_path = args.results_dir / "samples.json"
    samples_payload = {
        "class_names": train_dataset.classes,
        "records": records,
        "train_paths_recorded": len(train_paths),
    }
    samples_path.write_text(json.dumps(samples_payload, indent=2))

    print("Precision@k results:", json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved visualization metadata to {samples_path}")


if __name__ == "__main__":
    main()
