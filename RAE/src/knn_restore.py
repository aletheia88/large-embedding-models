from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from restore_methods import (
    fit_ridge_linear_map,
    linear_map_predict,
    evaluate_reconstruction,
    ResidualMLP,
)
import dataloader
import numpy as np
import random
import torch
import torch.nn.functional as F


def set_up():
    config_file = "stage1/pretrained/DINOv2-B.yaml"
    ckpt = "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    device = "cuda:0"
    batch_size = 64
    # seeds = [1912, 1985, 1976, 2001, 2024]
    seed = 1912
    num_workers = 1
    # schemes = ["baseline", "crop", "occlude", "gaussian"]
    scheme = "occlude"
    corrupt_range = (0.50, 0.70)
    k_neighbors = 25
    noise_std = None

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    global_configs = GlobalConfigs(
        config_file,
        ckpt,
        batch_size,
        k_neighbors,
        scheme,
        corrupt_range,
        noise_std,
        device,
        seed,
        num_workers,
        generator,
    )
    lam = None  # lam = 1 for "ridge" regressor method
    method = "mlp"  # options: "ridge", "mlp"
    hidden_dims = 128
    num_layers = 3
    dropout = 0.1
    cosine_weight = 1
    epochs = 1
    restore_configs = ReconstructConfigs(
        method,
        lam,
        hidden_dims,
        num_layers,
        dropout,
        cosine_weight,
        epochs,
    )

    return global_configs, restore_configs


@dataclass
class GlobalConfigs:
    config_file: str
    ckpt: str
    batch_size: int
    k_neighbors: int
    scheme: str
    corrupt_range: Optional[Tuple[float, float]]
    noise_std: Optional[float]
    device: str
    seed: int
    num_workers: int
    generator: torch.Generator
    config_root: Union[str, Path] = (
        "/home/alicialu/orcd/scratch/large-embedding-models/RAE/configs"
    )
    model_root: Union[str, Path] = (
        "/home/alicialu/orcd/scratch/large-embedding-models/RAE/models"
    )

    def __post_init__(self):
        # normalize roots
        self.config_root = Path(self.config_root).expanduser()
        self.model_root = Path(self.model_root).expanduser()

        # config path: if relative, join to root; if absolute, keep as-is
        cf = Path(self.config_file)
        self.config_path = cf if cf.is_absolute() else (self.config_root / cf)
        self.config_path = self.config_path.resolve()

        # model path: if relative, join to root; if absolute, keep as-is
        ck = Path(self.ckpt)
        self.ckpt_path = ck if ck.is_absolute() else (self.model_root / ck)
        self.ckpt_path = self.ckpt_path.resolve()


@dataclass
class ReconstructConfigs:
    method: str
    lam: Optional[float]
    hidden_dims: Optional[int]
    num_layers: Optional[int]
    dropout: Optional[float]
    cosine_weight: float
    epochs: Optional[int]


def get_embeddings_n_labels(configs, corruption=False):
    if corruption:
        scheme = configs.scheme
    else:
        scheme = "baseline"
    # dataloader for images
    train_loader, valid_loader = dataloader.get_imagenette_loaders(
        scheme, configs.corrupt_range, batch_size=configs.batch_size
    )
    rae = load_rae(configs.config_path, configs.ckpt_path, configs.device)
    train_embeddings, train_labels = extract_features(
        rae,
        train_loader,
        scheme,
        configs.device,
        generator=configs.generator,
        noise_std=configs.noise_std,
        max_items=None,
    )
    valid_embeddings, valid_labels = extract_features(
        rae,
        valid_loader,
        scheme,
        configs.device,
        generator=configs.generator,
        noise_std=configs.noise_std,
        max_items=None,
    )
    return {
        "embeddings": {"train": train_embeddings, "valid": valid_embeddings},
        "labels": {"train": train_labels, "valid": valid_labels},
    }


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
    noise_std: float,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std is None and scheme.lower() == "gaussian":
        raise ValueError(f"`noise_std` must be defined for the scheme {scheme}")

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
    train_feats,
    train_labels,
    test_feats,
    k,
    batch_size,
    device,
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


def reconstruct(global_configs: GlobalConfigs, restore_configs: ReconstructConfigs):
    clean_embeds_n_labels = get_embeddings_n_labels(global_configs)
    corrupt_embeds_n_labels = get_embeddings_n_labels(global_configs, corruption=True)

    clean_train_embeddings = clean_embeds_n_labels["embeddings"]["train"]
    corrupt_train_embeddings = corrupt_embeds_n_labels["embeddings"]["train"]
    clean_valid_embeddings = clean_embeds_n_labels["embeddings"]["valid"]
    corrupt_valid_embeddings = corrupt_embeds_n_labels["embeddings"]["valid"]
    train_labels = clean_embeds_n_labels["labels"]["train"]
    valid_labels = clean_embeds_n_labels["labels"]["valid"]

    if restore_configs.method == "ridge":
        W, b = fit_ridge_linear_map(
            X_train=corrupt_train_embeddings,
            Y_train=clean_train_embeddings,
            lam=restore_configs.lam,
        )
        train_embeds_pred = linear_map_predict(corrupt_train_embeddings, W, b)
        valid_embeds_pred = linear_map_predict(corrupt_valid_embeddings, W, b)

    if restore_configs.method == "mlp":
        # dataloader for image embeddings
        train_loader, val_loader = create_dataloaders(
            global_configs,
            restore_configs,
            clean_train_embeddings,
            corrupt_train_embeddings,
            clean_valid_embeddings,
            corrupt_valid_embeddings,
        )
        model, losses = train(
            restore_configs,
            train_loader,
            val_loader,
            global_configs.device,
        )
        model.eval()
        train_embeds_pred, valid_embeds_pred = evaluate(
            model, train_loader, val_loader, global_configs.device
        )
        print(f"pred train embeds: {train_embeds_pred.shape}")
        print(f"pred valid embeds: {valid_embeds_pred.shape}")

    # evaluation on train (sanity check)
    train_metrics = evaluate_reconstruction(train_embeds_pred, clean_train_embeddings)
    print("Train:", train_metrics)
    # validation reconstruction (the real test)
    valid_metrics = evaluate_reconstruction(valid_embeds_pred, clean_valid_embeddings)
    print("Valid:", valid_metrics)

    # see k-NN accuracy
    valid_labels_pred = knn_predict(
        train_embeds_pred,
        train_labels,
        valid_embeds_pred,
        k=global_configs.k_neighbors,
        batch_size=global_configs.batch_size,
        device=global_configs.device,
    )
    accuracy = (valid_labels_pred == valid_labels).float().mean().item()
    print(
        f"{global_configs.scheme} k-NN accuracy (k={global_configs.k_neighbors}): {accuracy * 100:.2f}%"
    )


def create_dataloaders(
    global_configs,
    restore_configs,
    clean_train_embeddings,
    corrupt_train_embeddings,
    clean_valid_embeddings,
    corrupt_valid_embeddings,
):
    # create dataloaders
    train_ds = TensorDataset(corrupt_train_embeddings, clean_train_embeddings)
    val_ds = TensorDataset(corrupt_valid_embeddings, clean_valid_embeddings)
    train_loader = DataLoader(
        train_ds, batch_size=global_configs.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=global_configs.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def train(
    restore_configs,
    train_loader,
    val_loader,
    device,
):
    in_dim = 768
    out_dim = 768
    model = ResidualMLP(
        in_dim,
        restore_configs.hidden_dims,
        out_dim,
        num_layers=restore_configs.num_layers,
        dropout=restore_configs.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    num_epochs = restore_configs.epochs
    cosine_weight = restore_configs.cosine_weight
    # training loop
    losses = {"train": [], "val": []}
    for epoch in tqdm(range(num_epochs)):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            # loss = MSE + cosine_weight * (1 - cosine_similiaty)
            mse_loss = F.mse_loss(y_pred, yb)
            cosine_loss = 1.0 - F.cosine_similarity(y_pred, yb, dim=-1, eps=1e-8).mean()
            loss = mse_loss + cosine_weight * cosine_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} loss: {loss.item()}")
            losses["train"].append(loss.item())

    return model, losses


def evaluate(model, train_loader, val_loader, device):
    yhat_train = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yhat_train.append(model(xb).cpu())
    train_pred = torch.cat(yhat_train, dim=0)

    yhat_val = []
    for xb, yb in val_loader:
        xb = xb.to(device)
        yhat_val.append(model(xb).cpu())
    val_pred = torch.cat(yhat_val, dim=0)

    return train_pred, val_pred


def main():
    global_configs, restore_configs = set_up()
    reconstruct(global_configs, restore_configs)


if __name__ == "__main__":
    main()
