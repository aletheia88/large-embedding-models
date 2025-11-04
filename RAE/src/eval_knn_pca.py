from pathlib import Path
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from tqdm import tqdm
import argparse
import imagenet_loader
import io
import json
import math
import torch
import torch.nn.functional as F


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
    rae.load_state_dict(state, strict=False)
    rae.eval()
    return rae


@torch.no_grad()
def extract_features(
    rae,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    n_seen = 0
    for images, y in tqdm(loader):
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


# @torch.no_grad()
# def knn_classify(
#     train_feats,
#     train_labels,
#     val_feats,
#     val_labels,
#     k: int,
#     t: float,
#     device: torch.device,
# ) -> float:
#     # Move memory bank to device
#     mem_x = train_feats.to(device)
#     mem_y = train_labels.to(device)
#     correct = 0
#     total = val_feats.size(0)
#     bs = 2048 // max(1, (train_feats.size(1) // 256))  # heuristic for batch compute
#     for i in range(0, total, bs):
#         q = val_feats[i : i + bs].to(device)  # [B, C]
#         sims = torch.mm(q, mem_x.t())  # [B, N]
#         topk = sims.topk(k, dim=1)
#         idx = topk.indices  # [B, k]
#         sim = topk.values / t
#         # gather labels and perform weighted voting
#         votes = torch.zeros(q.size(0), 1000, device=device, dtype=sim.dtype)
#         y_neighbors = mem_y[idx]
#         votes.scatter_add_(1, y_neighbors, sim.exp())
#         pred = votes.argmax(dim=1)
#         correct += (pred == val_labels[i : i + bs].to(device)).sum().item()
#     acc = correct / total * 100.0
#     return acc


@torch.no_grad()
def knn_classify(
    train_feats,
    train_labels,
    val_feats,
    val_labels,
    k: int,
    t: float,
    device: torch.device,
) -> float:
    # (Optional but usually important) normalize features for cosine-sim-like dot products
    train_feats = F.normalize(train_feats, dim=1)
    val_feats = F.normalize(val_feats, dim=1)

    # Move to device
    mem_x = train_feats.to(device)  # [N_train, C]
    mem_y = train_labels.to(device)  # [N_train]

    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)
    print("train label range:", train_labels.min().item(), train_labels.max().item())
    print("val label range:", val_labels.min().item(), val_labels.max().item())
    return
    total = val_feats.size(0)
    correct = 0

    # Infer number of classes from labels instead of hard-coding 1000
    num_classes = int(mem_y.max().item() + 1)

    # Heuristic batch size for computing sim matrix in chunks
    bs = 2048 // max(1, (train_feats.size(1) // 256))

    for i in range(0, total, bs):
        q = val_feats[i : i + bs]  # [B, C]
        sims = torch.mm(q, mem_x.t())  # [B, N_train]

        # Top-k nearest neighbors
        topk = sims.topk(k, dim=1)
        idx = topk.indices  # [B, k]
        sim = topk.values / t  # temperature scaling

        # Neighbor labels: [B, k]
        y_neighbors = mem_y[idx]

        # Weighted voting over classes
        votes = torch.zeros(q.size(0), num_classes, device=device, dtype=sim.dtype)
        weights = sim.exp()  # [B, k]
        votes.scatter_add_(
            1, y_neighbors, weights
        )  # accumulate weights into class bins

        # Predicted class is argmax of votes
        pred = votes.argmax(dim=1)  # [B]
        correct += (pred == val_labels[i : i + bs]).sum().item()

    acc = correct / total * 100.0
    return acc


def pca_compress(X: torch.Tensor, n_components: int):
    """
    Project X onto PCs. If whiten=True, divide by sqrt(variance).

    X: [N, D] float tensor (N samples, D features)
    k: target dim (k <= D)

    Returns: mean [D], components [D, k], explained_var [k]
    """
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False)
    Z = pca.fit_transform(X.numpy())
    return Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-t", type=float, default=0.07)
    parser.add_argument("--pca-n", type=int, default=512)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on number of train samples for faster kNN.",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional cap on number of val samples for faster kNN.",
    )
    args = parser.parse_args()
    # batch_size = 256
    # num_workers = 1
    # device = "cuda:0"
    # k-NN params
    # t = 0.07
    # k = 20
    data_root = Path("/home/alicialu/orcd/scratch/imagenet")
    paths = imagenet_loader.ImageNetPaths(
        train_dir=data_root / "train",  # contains 1000 *.tar shards
        val_tar=data_root / "ILSVRC2012_img_val.tar",
        devkit_dir=data_root / "ILSVRC2012_devkit_t12",
    )
    print("Building loaders...")
    train_loader, val_loader, wnid_to_idx = imagenet_loader.build_imagenet_loaders(
        paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
        augment_train=True,
    )
    print("Done.")

    config_root = "/home/alicialu/orcd/scratch/large-embedding-models/RAE/configs"
    # config_path = f"{config_root}/stage1/pretrained/DINOv2-B.yaml"
    config_path = f"{config_root}/{args.config}"
    model_root = "/home/alicialu/orcd/scratch/large-embedding-models/RAE/models"
    # ckpt_path = f"{model_root}/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    ckpt_path = f"{model_root}/{args.ckpt}"
    print("Loading RAE...")
    rae = load_rae(config_path, ckpt_path, args.device)
    print("Done.")

    print("Extracting features...")
    train_feats, train_labels = extract_features(
        rae,
        train_loader,
        args.device,
        max_items=args.max_train,
    )
    # (original) features: torch.Size([batch_size, 768])
    # labels: torch.Size([batch_size])
    valid_feats, valid_labels = extract_features(
        rae,
        val_loader,
        args.device,
        max_items=args.max_val,
    )
    print(f"features: {train_feats.shape} {valid_feats.shape}")
    print(f"labels: {train_labels.shape}")
    print("Done.")

    print("Compressing features...")
    pca_train_feats = pca_compress(train_feats, n_components=args.pca_n)
    pca_valid_feats = pca_compress(valid_feats, n_components=args.pca_n)
    print(f"pca features: {pca_train_feats.shape} {pca_valid_feats.shape}")

    print("K-NN clustering...")

    knn_accuracy = knn_classify(
        train_feats,
        train_labels,
        valid_feats,
        valid_labels,
        args.knn_k,
        args.knn_t,
        args.device,
    )
    print(f"knn accuracy: {knn_accuracy}")


if __name__ == "__main__":
    main()
