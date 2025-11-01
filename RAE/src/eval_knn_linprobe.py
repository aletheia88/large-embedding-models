import argparse
import io
import json
import math
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from PIL import Image

from omegaconf import OmegaConf
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config


class ImageNetTarDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: List[Path],
        transform: transforms.Compose,
        shuffle: bool = False,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.transform = transform
        self.shuffle = shuffle
        self.base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _iter_shard(self, shard_path: Path, rng: torch.Generator):
        with tarfile.open(shard_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            sample_ids = [name[:-4] for name in members if name.endswith(".jpg")]
            if self.shuffle:
                # torch.Generator-based shuffle for worker-determinism
                perm = torch.randperm(len(sample_ids), generator=rng).tolist()
                sample_ids = [sample_ids[i] for i in perm]
            for sid in sample_ids:
                img_mem = members.get(f"{sid}.jpg")
                cls_mem = members.get(f"{sid}.cls")
                if img_mem is None or cls_mem is None:
                    continue
                img_f = tar.extractfile(img_mem)
                cls_f = tar.extractfile(cls_mem)
                if img_f is None or cls_f is None:
                    continue
                try:
                    label = int(cls_f.read().decode("utf-8").strip())
                except Exception:
                    continue
                with Image.open(io.BytesIO(img_f.read())) as pil:
                    img = pil.convert("RGB")
                yield self.transform(img), label

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        # shard worker split
        shard_indices = list(range(len(self.shard_paths)))[worker_id::num_workers]
        rng = torch.Generator()
        rng.manual_seed(self.base_seed + 7919 * (self._epoch + 1) + worker_id)
        for idx in shard_indices:
            yield from self._iter_shard(self.shard_paths[idx], rng)


@dataclass
class EvalArgs:
    config: str
    ckpt: str
    data_root: str
    batch_size: int
    num_workers: int
    device: str
    results_dir: str
    knn_k: int
    knn_t: float
    lin_epochs: int
    lin_lr: float


def build_eval_loaders(
    data_root: str,
    batch_size: int,
    workers: int,
    train_aug: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(data_root)
    train_shards = sorted(root.glob("imagenet1k-train-*.tar"))
    val_shards = sorted(root.glob("imagenet1k-validation-*.tar"))
    if not train_shards or not val_shards:
        raise FileNotFoundError("Did not find train/validation WebDataset shards under data_root.")

    # Standard eval preprocessing: 256 resize -> 224 center crop
    tfm_eval = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    if train_aug:
        tfm_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        tfm_train = tfm_eval

    ds_train = ImageNetTarDataset(train_shards, tfm_train, shuffle=train_aug)
    ds_val = ImageNetTarDataset(val_shards, tfm_eval, shuffle=False)
    loader_train = DataLoader(ds_train, batch_size=batch_size, num_workers=workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, pin_memory=True)
    return loader_train, loader_val


def load_rae(config_path: str, ckpt_path: str, device: torch.device):
    (stage1_cfg, *_rest) = parse_configs(config_path)
    # Ensure DINOv3 encoder keeps its final norm affine params for eval
    try:
        if stage1_cfg.params.encoder_cls in ("Dinov3withNorm", "Dinov3WithNorm"):
            stage1_cfg.params.encoder_params.normalize = False
    except Exception:
        pass
    rae = instantiate_from_config(stage1_cfg).to(device)
    rae.eval()
    # load model weights
    state = torch.load(ckpt_path, map_location="cpu")
    rae.load_state_dict(state["model"], strict=False)
    rae.eval()
    return rae


@torch.no_grad()
def extract_features(rae, loader: DataLoader, device: torch.device, max_items: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    n_seen = 0
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        z = rae.encode(images)  # B x C x H x W
        z = z.mean(dim=(-2, -1))  # GAP -> B x C
        feats.append(F.normalize(z, dim=-1).cpu())
        labels.append(y.cpu())
        n_seen += images.size(0)
        if max_items is not None and n_seen >= max_items:
            break
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


@torch.no_grad()
def knn_classify(train_feats, train_labels, val_feats, val_labels, k: int, t: float, device: torch.device) -> float:
    # Move memory bank to device
    mem_x = train_feats.to(device)
    mem_y = train_labels.to(device)
    correct = 0
    total = val_feats.size(0)
    bs = 2048 // max(1, (train_feats.size(1) // 256))  # heuristic for batch compute
    for i in range(0, total, bs):
        q = val_feats[i : i + bs].to(device)  # [B, C]
        sims = torch.mm(q, mem_x.t())  # [B, N]
        topk = sims.topk(k, dim=1)
        idx = topk.indices  # [B, k]
        sim = topk.values / t
        # gather labels and perform weighted voting
        votes = torch.zeros(q.size(0), 1000, device=device, dtype=sim.dtype)
        y_neighbors = mem_y[idx]
        votes.scatter_add_(1, y_neighbors, sim.exp())
        pred = votes.argmax(dim=1)
        correct += (pred == val_labels[i : i + bs].to(device)).sum().item()
    acc = correct / total * 100.0
    return acc


class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 1000):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(rae, loader_train, loader_val, device: torch.device, epochs: int = 10, lr: float = 0.1) -> Tuple[float, float]:
    rae.eval()
    for p in rae.parameters():
        p.requires_grad = False
    head = LinearHead(rae.latent_dim, 1000).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    def eval_loader(loader) -> float:
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, y in loader:
                images = images.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                z = rae.encode(images).mean(dim=(-2, -1))
                logits = head(z)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return 100.0 * correct / total

    for epoch in range(epochs):
        head.train()
        running = 0.0
        n = 0
        for images, y in loader_train:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                z = rae.encode(images).mean(dim=(-2, -1))
                logits = head(z)
                loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        sched.step()
        # optional: print epoch summary
        print(f"[LinearProbe] Epoch {epoch+1}/{epochs} loss={running/n:.4f}")

    # final eval
    train_acc = eval_loader(loader_train)
    val_acc = eval_loader(loader_val)
    return train_acc, val_acc


def train_linear_probe_from_feats(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.1,
    batch_size: int = 4096,
) -> Tuple[float, float]:
    num_classes = int(train_labels.max().item() + 1)
    in_dim = train_feats.size(1)
    head = LinearHead(in_dim, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    tr_x = train_feats.to(device=device, dtype=torch.float32)
    tr_y = train_labels.to(device)
    va_x = val_feats.to(device=device, dtype=torch.float32)
    va_y = val_labels.to(device)

    N = tr_x.size(0)
    order = torch.arange(N, device=device)

    def eval_split(x, y) -> float:
        head.eval()
        with torch.no_grad():
            logits = head(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item() * 100.0
        return acc

    for ep in range(epochs):
        head.train()
        # simple shuffling on device
        perm = order[torch.randperm(N, device=device)]
        running = 0.0
        seen = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            xb = tr_x[idx]
            yb = tr_y[idx]
            logits = head(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        sched.step()
        print(f"[LinearProbe-Feats] Epoch {ep+1}/{epochs} loss={running/seen:.4f}")

    train_acc = eval_split(tr_x, tr_y)
    val_acc = eval_split(va_x, va_y)
    return train_acc, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-t", type=float, default=0.07)
    parser.add_argument("--lin-epochs", type=int, default=10)
    parser.add_argument("--lin-lr", type=float, default=0.1)
    parser.add_argument("--skip-linprobe", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--linear-online", action="store_true", help="Train linear head on-the-fly with train augmentations.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on number of train samples for faster kNN.")
    parser.add_argument("--max-val", type=int, default=None, help="Optional cap on number of val samples for faster kNN.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    print("Building loaders…")
    loader_train, loader_val = build_eval_loaders(
        args.data_root, args.batch_size, args.num_workers, train_aug=args.linear_online
    )
    print("Loading model…")
    rae = load_rae(args.config, args.ckpt, device)

    if args.linear_online:
        # For online linear probe, we still need val features for evaluation
        print("Extracting validation features…")
        val_feats, val_labels = extract_features(rae, loader_val, device, max_items=args.max_val)
        print(f"Val feats: {val_feats.shape}, labels: {val_labels.shape}")
        # For kNN or feature-based linear probe, also extract train features
        if not args.skip_knn:
            print("Extracting train features for k-NN…")
            train_feats, train_labels = extract_features(rae, loader_train, device, max_items=args.max_train)
            print(f"Train feats: {train_feats.shape}, labels: {train_labels.shape}")
    else:
        print("Extracting train features…")
        train_feats, train_labels = extract_features(rae, loader_train, device, max_items=args.max_train)
        print(f"Train feats: {train_feats.shape}, labels: {train_labels.shape}")
        print("Extracting val features…")
        val_feats, val_labels = extract_features(rae, loader_val, device, max_items=args.max_val)
        print(f"Val feats: {val_feats.shape}, labels: {val_labels.shape}")

    results = {}
    if not args.skip_knn:
        print("Running k-NN…")
        knn_acc = knn_classify(train_feats, train_labels, val_feats, val_labels, args.knn_k, args.knn_t, device)
        print(f"k-NN (k={args.knn_k}, T={args.knn_t}) top-1 acc: {knn_acc:.2f}")
        results["knn_top1"] = knn_acc

    if not getattr(args, "skip_linprobe", False):
        print("Training linear probe…")
        if args.linear_online:
            tr_acc, val_acc = train_linear_probe(rae, loader_train, loader_val, device, args.lin_epochs, args.lin_lr)
        else:
            tr_acc, val_acc = train_linear_probe_from_feats(
                train_feats, train_labels, val_feats, val_labels, device, args.lin_epochs, args.lin_lr
            )
        print(f"Linear probe top-1: train={tr_acc:.2f}, val={val_acc:.2f}")
        results.update({"linprobe_train_top1": tr_acc, "linprobe_val_top1": val_acc})

    with open(Path(args.results_dir) / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", Path(args.results_dir) / "eval_results.json")


if __name__ == "__main__":
    main()
