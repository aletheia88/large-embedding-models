import argparse
import io
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

from feature_mae.model import FeatureMaskedAutoencoder
from feature_mae.utils import FeatureNormalizer
from stage1.encoders.dinov2 import Dinov2withNorm


class ImageNetTarDataset(IterableDataset):
    def __init__(self, shards: List[Path], transform: transforms.Compose, shuffle: bool = False, seed: int = 0):
        super().__init__()
        self.shards = [Path(s) for s in shards]
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        indices = list(range(wid, len(self.shards), nw))
        g = torch.Generator()
        g.manual_seed(self.seed + 1337 + wid)
        for idx in indices:
            with tarfile.open(self.shards[idx], "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                ids = [n[:-4] for n in members if n.endswith(".jpg")]
                if self.shuffle:
                    perm = torch.randperm(len(ids), generator=g).tolist()
                    ids = [ids[i] for i in perm]
                for sid in ids:
                    im = members.get(f"{sid}.jpg")
                    cl = members.get(f"{sid}.cls")
                    if im is None or cl is None:
                        continue
                    fi = tar.extractfile(im)
                    fl = tar.extractfile(cl)
                    if fi is None or fl is None:
                        continue
                    label = int(fl.read().decode("utf-8").strip())
                    with Image.open(io.BytesIO(fi.read())) as pil:
                        img = pil.convert("RGB")
                    yield self.transform(img), label


def build_loaders(root: str, batch_size: int, workers: int, center_crop: bool = True) -> Tuple[DataLoader, DataLoader]:
    root = Path(root)
    train = sorted(root.glob("imagenet1k-train-*.tar"))
    val = sorted(root.glob("imagenet1k-validation-*.tar"))
    if not train or not val:
        raise FileNotFoundError("Missing ImageNet WebDataset shards.")
    if center_crop:
        tfm_train = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2,1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    tfm_val = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    ds_train = ImageNetTarDataset(train, tfm_train, shuffle=not center_crop)
    ds_val = ImageNetTarDataset(val, tfm_val, shuffle=False)
    ld_train = DataLoader(ds_train, batch_size=batch_size, num_workers=workers, pin_memory=True)
    ld_val = DataLoader(ds_val, batch_size=batch_size, num_workers=workers, pin_memory=True)
    return ld_train, ld_val


@torch.no_grad()
def extract_feats_dinov2(encoder: Dinov2withNorm, loader: DataLoader, device: torch.device, max_items: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    n = 0
    for images, y in loader:
        images = images.to(device)
        # Dinov2withNorm forward returns patch tokens (excludes CLS + registers)
        toks = encoder(images)  # [B, N, C]
        vec = F.normalize(toks.mean(dim=1), dim=-1)
        xs.append(vec.cpu())
        ys.append(y)
        n += images.size(0)
        if max_items is not None and n >= max_items:
            break
    return torch.cat(xs, 0), torch.cat(ys, 0)


@torch.no_grad()
def extract_feats_feature_mae(
    dino: Dinov2withNorm,
    fmae: FeatureMaskedAutoencoder,
    loader: DataLoader,
    device: torch.device,
    normalizer: Optional[FeatureNormalizer] = None,
    mask_ratio: float = 0.0,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    n = 0
    for images, y in loader:
        images = images.to(device)
        toks = dino(images)  # [B, N, C]
        if normalizer is not None:
            toks = normalizer.normalize(toks)
        pred, target, _ = fmae(toks, mask_ratio=mask_ratio)
        # Use reconstructed tokens (pred) as features
        rec = pred  # [B, N, C]
        if normalizer is not None:
            rec = normalizer.denormalize(rec)
        vec = F.normalize(rec.mean(dim=1), dim=-1)
        xs.append(vec.cpu())
        ys.append(y)
        n += images.size(0)
        if max_items is not None and n >= max_items:
            break
    return torch.cat(xs, 0), torch.cat(ys, 0)


@torch.no_grad()
def knn(mem_x: torch.Tensor, mem_y: torch.Tensor, qry_x: torch.Tensor, qry_y: torch.Tensor, k: int = 20, t: float = 0.07, device: torch.device = torch.device('cuda')) -> float:
    mem_x = mem_x.to(device)
    mem_y = mem_y.to(device)
    total = qry_x.size(0)
    correct = 0
    step = 2048
    for i in range(0, total, step):
        q = qry_x[i:i+step].to(device)
        sims = torch.mm(q, mem_x.t())
        topk = sims.topk(k, dim=1)
        idx = topk.indices
        sim = (topk.values / t).exp()
        votes = torch.zeros(q.size(0), 1000, device=device, dtype=sim.dtype)
        votes.scatter_add_(1, mem_y[idx], sim)
        pred = votes.argmax(dim=1)
        correct += (pred == qry_y[i:i+step].to(device)).sum().item()
    return correct / total * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--featuremae-ckpt', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-train', type=int, default=None)
    ap.add_argument('--max-val', type=int, default=None)
    ap.add_argument('--knn-k', type=int, default=20)
    ap.add_argument('--knn-t', type=float, default=0.07)
    ap.add_argument('--use-center-crop', action='store_true')
    ap.add_argument('--use-normalizer', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ld_train, ld_val = build_loaders(args.data_root, args.batch_size, args.num_workers, center_crop=args.use_center_crop)

    # Build raw DINOv2 encoder (keep final LN affine for eval)
    dino = Dinov2withNorm(dinov2_path='facebook/dinov2-with-registers-base', normalize=False).to(device)
    dino.eval()

    print('Extracting DINOv2 features...')
    tr_x, tr_y = extract_feats_dinov2(dino, ld_train, device, args.max_train)
    va_x, va_y = extract_feats_dinov2(dino, ld_val, device, args.max_val)
    print(f'DINOv2 feats: train {tr_x.shape}, val {va_x.shape}')
    acc_dino = knn(tr_x, tr_y, va_x, va_y, k=args.knn_k, t=args.knn_t, device=device)
    print(f'DINOv2 k-NN top-1: {acc_dino:.2f}')

    # Build FeatureMAE
    feat_dim = dino.hidden_size
    num_patches = (224 // dino.patch_size) ** 2
    fmae = FeatureMaskedAutoencoder(
        feature_dim=feat_dim,
        num_patches=num_patches,
        encoder_embed_dim=feat_dim,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    )
    state = torch.load(args.featuremae_ckpt, map_location='cpu')
    missing = fmae.load_state_dict(state['model'], strict=False)
    if isinstance(missing, torch.nn.modules.module._IncompatibleKeys):
        if missing.missing_keys:
            print('Missing keys:', missing.missing_keys)
        if missing.unexpected_keys:
            print('Unexpected keys:', missing.unexpected_keys)
    fmae.to(device).eval()

    normalizer = FeatureNormalizer('models/stats/dinov2/wReg_base/imagenet1k/stat.pt').to(device) if args.use_normalizer else None

    print('Extracting FeatureMAE features...')
    tr_fx, tr_fy = extract_feats_feature_mae(dino, fmae, ld_train, device, normalizer=normalizer, mask_ratio=0.0, max_items=args.max_train)
    va_fx, va_fy = extract_feats_feature_mae(dino, fmae, ld_val, device, normalizer=normalizer, mask_ratio=0.0, max_items=args.max_val)
    print(f'FeatureMAE feats: train {tr_fx.shape}, val {va_fx.shape}')
    acc_fmae = knn(tr_fx, tr_fy, va_fx, va_fy, k=args.knn_k, t=args.knn_t, device=device)
    print(f'FeatureMAE k-NN top-1: {acc_fmae:.2f}')


if __name__ == '__main__':
    main()

