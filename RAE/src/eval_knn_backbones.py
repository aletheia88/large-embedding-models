import argparse
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

from feature_mae.model import FeatureMaskedAutoencoder
from feature_mae.utils import FeatureNormalizer


class ImageNetTarDataset(IterableDataset):
    def __init__(self, shards: List[Path], transform: transforms.Compose, shuffle: bool = False, seed: int = 0):
        super().__init__()
        self.shards = [Path(s) for s in shards]
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        import tarfile
        import random

        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        shard_idxs = list(range(len(self.shards)))[worker_id::num_workers]
        rng = random.Random(self.seed + 7919 * (self._epoch + 1) + worker_id)
        for si in shard_idxs:
            with tarfile.open(self.shards[si], "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                sample_ids = [n[:-4] for n in members if n.endswith(".jpg")]
                if self.shuffle:
                    rng.shuffle(sample_ids)
                for sid in sample_ids:
                    im = members.get(f"{sid}.jpg"); cl = members.get(f"{sid}.cls")
                    if im is None or cl is None:
                        continue
                    imf = tar.extractfile(im); clf = tar.extractfile(cl)
                    if imf is None or clf is None:
                        continue
                    y = int(clf.read().decode("utf-8").strip())
                    with Image.open(io.BytesIO(imf.read())) as pil:
                        x = pil.convert("RGB")
                    yield self.transform(x), y


def build_loaders(data_root: str, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    root = Path(data_root)
    train_shards = sorted(root.glob("imagenet1k-train-*.tar"))
    val_shards = sorted(root.glob("imagenet1k-validation-*.tar"))
    if not train_shards or not val_shards:
        raise FileNotFoundError("Could not find train/validation shards under data_root")
    tfm_eval = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds_train = ImageNetTarDataset(train_shards, tfm_eval, shuffle=False)
    ds_val = ImageNetTarDataset(val_shards, tfm_eval, shuffle=False)
    ltr = DataLoader(ds_train, batch_size=batch_size, num_workers=workers, pin_memory=True)
    lva = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, pin_memory=True)
    return ltr, lva


@torch.no_grad()
def extract_dinov2_feats(model: Dinov2WithRegistersModel, proc, loader: DataLoader, device: torch.device, max_items: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    seen = 0
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        x = (x - mean) / std
        out = model(x, output_hidden_states=True)
        tok = out.last_hidden_state[:, 5:, :]  # drop 1 CLS + 4 registers
        v = tok.mean(dim=1)  # GAP over tokens
        v = F.normalize(v, dim=-1).cpu()
        feats.append(v)
        labels.append(y)
        seen += x.size(0)
        if max_items is not None and seen >= max_items:
            break
    return torch.cat(feats, 0), torch.cat(labels, 0)


def build_feature_mae_from_ckpt(ckpt_path: str, feature_dim: int, num_tokens: int, device: torch.device) -> Tuple[FeatureMaskedAutoencoder, Optional[FeatureNormalizer]]:
    state = torch.load(ckpt_path, map_location="cpu")
    args = state.get("args", {})
    # If checkpoint stores positional embedding, infer token count from it for shape compatibility
    try:
        num_tokens = int(state["model"]["encoder_pos_embed"].shape[1] - 1)
    except Exception:
        pass
    enc_dim = int(args.get("encoder_embed_dim", feature_dim))
    model = FeatureMaskedAutoencoder(
        feature_dim=feature_dim,
        num_tokens=num_tokens,
        encoder_embed_dim=enc_dim,
        encoder_depth=int(args.get("encoder_depth", 12)),
        encoder_num_heads=int(args.get("encoder_heads", 12)),
        decoder_embed_dim=int(args.get("decoder_embed_dim", 512)),
        decoder_depth=int(args.get("decoder_depth", 8)),
        decoder_num_heads=int(args.get("decoder_heads", 16)),
    ).to(device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[FeatureMAE] Incompatible keys:", incompatible)
    model.eval()
    norm = None
    stat_path = args.get("feature_stat_path", None)
    if stat_path and Path(stat_path).exists():
        norm = FeatureNormalizer(stat_path).to(device)
    return model, norm


@torch.no_grad()
def extract_featuremae_feats(
    dino: Dinov2WithRegistersModel,
    proc,
    featmae: FeatureMaskedAutoencoder,
    normalizer: Optional[FeatureNormalizer],
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    seen = 0
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    expected_tokens = int(featmae.encoder_pos_embed.shape[1] - 1)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        x = (x - mean) / std
        out = dino(x, output_hidden_states=True)
        tok_full = out.last_hidden_state  # may include CLS + registers
        n_full = tok_full.size(1)
        # Use the tail tokens to align to image tokens when expected excludes specials; otherwise use full
        if expected_tokens == n_full:
            tok = tok_full
        elif expected_tokens == n_full - 5:
            tok = tok_full[:, 5:, :]
        else:
            tok = tok_full[:, :expected_tokens, :]
        if normalizer is not None:
            b, n, c = tok.shape
            h = int(n ** 0.5)
            if h * h == n:
                hw = tok.transpose(1, 2).reshape(b, c, h, h)
                hw = normalizer.normalize(hw.transpose(1, 2).reshape(b, n, c))
                tok = hw
        # encoder pass with no masking
        xenc = tok + featmae.encoder_pos_embed[:, 1:, :]
        cls_tok = featmae.encoder_cls_token + featmae.encoder_pos_embed[:, :1, :]
        cls_tok = cls_tok.expand(xenc.size(0), -1, -1)
        xseq = torch.cat([cls_tok, xenc], dim=1)
        xseq = featmae.encoder(xseq)
        xseq = featmae.encoder_norm(xseq)
        cls = xseq[:, 0, :]
        v = F.normalize(cls, dim=-1).cpu()
        feats.append(v)
        labels.append(y)
        seen += x.size(0)
        if max_items is not None and seen >= max_items:
            break
    return torch.cat(feats, 0), torch.cat(labels, 0)


@torch.no_grad()
def knn(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, k: int, t: float, device: torch.device) -> float:
    mem_x = train_x.to(device)
    mem_y = train_y.to(device)
    total = val_x.size(0)
    correct = 0
    bs = 512
    for i in range(0, total, bs):
        q = val_x[i : i + bs].to(device)
        sims = torch.mm(q, mem_x.t())
        topk = sims.topk(k, dim=1)
        idx = topk.indices
        sim = topk.values / t
        votes = torch.zeros(q.size(0), 1000, device=device, dtype=sim.dtype)
        votes.scatter_add_(1, mem_y[idx], sim.exp())
        pred = votes.argmax(dim=1)
        correct += (pred == val_y[i : i + bs].to(device)).sum().item()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dinov2", "featuremae"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-t", type=float, default=0.07)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--dinov2-id", default="facebook/dinov2-with-registers-base")
    ap.add_argument("--featuremae-ckpt", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(args.dinov2_id)
    loader_tr, loader_va = build_loaders(args.data_root, args.batch_size, args.num_workers)

    print("Loading DINOv2…")
    dino = Dinov2WithRegistersModel.from_pretrained(args.dinov2_id)
    dino.to(device).eval().requires_grad_(False)

    if args.mode == "dinov2":
        tr_x, tr_y = extract_dinov2_feats(dino, proc, loader_tr, device, args.max_train)
        va_x, va_y = extract_dinov2_feats(dino, proc, loader_va, device, args.max_val)
        acc = knn(tr_x, tr_y, va_x, va_y, args.knn_k, args.knn_t, device)
        print(f"DINOv2 k-NN top-1: {acc:.2f}")
        with open(Path(args.results_dir) / "dinov2_knn.json", "w") as f:
            json.dump({"knn_top1": acc}, f, indent=2)
    else:
        if not args.featuremae_ckpt:
            raise ValueError("--featuremae-ckpt is required for mode=featuremae")
        # compute meta
        patch = dino.config.patch_size
        num_tokens = (224 // patch) ** 2
        featmae, norm = build_feature_mae_from_ckpt(args.featuremae_ckpt, dino.config.hidden_size, num_tokens, device)
        tr_x, tr_y = extract_featuremae_feats(dino, proc, featmae, norm, loader_tr, device, args.max_train)
        va_x, va_y = extract_featuremae_feats(dino, proc, featmae, norm, loader_va, device, args.max_val)
        acc = knn(tr_x, tr_y, va_x, va_y, args.knn_k, args.knn_t, device)
        print(f"FeatureMAE k-NN top-1: {acc:.2f}")
        with open(Path(args.results_dir) / "featuremae_knn.json", "w") as f:
            json.dump({"knn_top1": acc}, f, indent=2)


if __name__ == "__main__":
    main()
