from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Tuple
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
import dataloader
import numpy as np
import random
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
    scheme: str,
    device: torch.device,
    generator: torch.Generator,
    noise_std: float = None,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std is None and scheme.lower() == "gaussian":
        raise ValueError(f"`noise_std` must be defined for the scheme {{scheme}}")

    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    n_seen = 0
    for images, y in tqdm(loader):
        images = images.to(device, non_blocking=True)

        z = rae.encode(images)  # (B, C, H, W)
        z = z.mean(dim=(-2, -1))  # (B, C)
        z = F.normalize(z, dim=-1)  # (B, C), unit norm

        if scheme.lower() == "gaussian" and noise_std > 0:
            eps = torch.randn(
                z.shape, device=z.device, dtype=z.dtype, generator=generator
            )
            z = z + noise_std * eps
            # renormalize after noise
            z = F.normalize(z, dim=-1)

        feats.append(z.cpu())
        labels.append(y.cpu())

        n_seen += images.size(0)
        if max_items is not None and n_seen >= max_items:
            break

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    return feats, labels


def knn_predict(
    train_feats, train_labels, test_feats, k=5, batch_size=512, device="cuda"
):
    """
    train_feats: [N_train, D]
    train_labels: [N_train]
    test_feats: [N_test, D]
    """
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    preds_all = []

    with torch.no_grad():
        for start in range(0, test_feats.size(0), batch_size):
            end = start + batch_size
            x = test_feats[start:end]  # [B, D]

            # Squared L2 distance between x and all train_feats
            # x: [B,1,D], train: [1,N,D] -> [B,N,D] -> [B,N]
            diff = x.unsqueeze(1) - train_feats.unsqueeze(0)
            dists = (diff**2).sum(dim=2)  # [B, N_train]

            # indices of k smallest distances
            knn_inds = dists.topk(k, largest=False).indices  # [B, k]

            # labels of those neighbors
            knn_labels = train_labels[knn_inds]  # [B, k]

            # majority vote
            pred = torch.mode(knn_labels, dim=1).values  # [B]
            preds_all.append(pred.cpu())

    return torch.cat(preds_all, dim=0)  # [N_test]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    # make Python/numpy in each worker deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    config_root = "/home/alicialu/orcd/scratch/large-embedding-models/RAE/configs"
    config_path = f"{config_root}/stage1/pretrained/DINOv2-B.yaml"
    model_root = "/home/alicialu/orcd/scratch/large-embedding-models/RAE/models"
    ckpt_path = f"{model_root}/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    device = "cuda:0"

    batch_size = 64
    # seeds = [1912, 1985, 1976, 2001, 2024]
    seeds = [1912]
    # schemes = ["baseline", "crop", "occlude", "gaussian"]
    schemes = ["gaussian"]
    corrupt_range = (0.50, 0.70)
    k_neighbors = 25
    noise_std = 0.01
    for seed in seeds:
        for scheme in schemes:
            seed_everything(seed)
            g = torch.Generator(device=device).manual_seed(seed)
            train_loader, valid_loader = dataloader.get_imagenette_loaders(
                scheme, corrupt_range, batch_size=batch_size
            )
            print("dataloder built.")
            rae = load_rae(config_path, ckpt_path, device)
            print("model built.")
            train_features, train_labels = extract_features(
                rae,
                train_loader,
                scheme,
                device,
                generator=g,
                noise_std=noise_std,
                max_items=None,
            )
            valid_features, valid_labels = extract_features(
                rae,
                valid_loader,
                scheme,
                device,
                generator=g,
                noise_std=noise_std,
                max_items=None,
            )
            valid_preds = knn_predict(
                train_features,
                train_labels,
                valid_features,
                k=k_neighbors,
                batch_size=batch_size,
                device=device,
            )
            accuracy = (valid_preds == valid_labels).float().mean().item()
            print(f"{scheme} k-NN accuracy (k={k_neighbors}): {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
