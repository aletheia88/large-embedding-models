import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dataloader
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableUnCLIPImg2ImgPipeline
from restore_methods import (
    ResidualMLP,
    HopfieldLayerDecoder,
    HopfieldAssociateDecoder,
    evaluate_reconstruction,
    fit_ridge_linear_map,
    linear_map_predict,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.transforms.functional import to_pil_image


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def set_up():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    # seeds = [1912, 1985, 1976, 2001, 2024]
    seed = 1912
    num_workers = 2
    # schemes = ["baseline", "crop", "occlude", "gaussian"]
    scheme = "mix"  # options: "baseline", "occlude", "mix", ...
    corrupt_range = (0.50, 0.70)
    k_neighbors = 25
    noise_std = None
    mix_alpha = 0.5

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    global_configs = GlobalConfigs(
        batch_size=batch_size,
        k_neighbors=k_neighbors,
        scheme=scheme,
        corrupt_range=corrupt_range,
        noise_std=noise_std,
        device=device,
        seed=seed,
        num_workers=num_workers,
        generator=generator,
        mix_alpha=mix_alpha,
    )
    # lam = 1 for "ridge" regressor method
    lam = None
    # options: "ridge", "mlp", "hopfield", "hopfield-associate"
    method = "mlp"
    hidden_dims = 2048
    num_layers = 3
    dropout = 0.1
    epochs = 5
    restore_configs = ReconstructConfigs(
        method,
        lam,
        hidden_dims,
        num_layers,
        dropout,
        epochs,
    )

    encoder_type = "clip"  # options: "dino", "clip"
    if encoder_type == "clip":
        encoder_model_name = "sd2-community/stable-diffusion-2-1-unclip-small"
        model_subfolder = "image_encoder"
        processor_subfolder = "feature_extractor"
        normalize_features = False
    else:
        encoder_model_name = "facebook/dinov2-base"
        model_subfolder = None
        processor_subfolder = None
        normalize_features = True

    print("Building Encoder configs...")
    encoder_configs = EncoderConfigs(
        encoder_type=encoder_type,
        model_name=encoder_model_name,
        image_size=224,
        model_subfolder=model_subfolder,
        processor_subfolder=processor_subfolder,
        normalize_features=normalize_features,
    )

    unclip_configs = UnCLIPConfigs(
        enabled=encoder_type == "clip",
        model_id="sd2-community/stable-diffusion-2-1-unclip-small",
        output_dir=os.path.join("results", "unclip_valid"),
        num_images=16,
        batch_size=4,
        guidance_scale=5.0,
        num_inference_steps=30,
        torch_dtype="fp16" if "cuda" in device else "fp32",
        seed=seed,
        enable_attention_slicing=True,
    )

    return global_configs, encoder_configs, restore_configs, unclip_configs


@dataclass
class GlobalConfigs:
    batch_size: int
    k_neighbors: int
    scheme: str
    corrupt_range: Optional[Tuple[float, float]]
    noise_std: Optional[float]
    device: str
    seed: int
    num_workers: int
    generator: torch.Generator
    mix_alpha: Optional[float] = None


@dataclass
class ReconstructConfigs:
    method: str
    lam: Optional[float]
    hidden_dims: Optional[int]
    num_layers: Optional[int]
    dropout: Optional[float]
    epochs: Optional[int]


@dataclass
class EncoderConfigs:
    encoder_type: str  # "dino" or "clip"
    model_name: str
    image_size: int = 224
    model_subfolder: Optional[str] = None
    processor_subfolder: Optional[str] = None
    normalize_features: bool = True


@dataclass
class UnCLIPConfigs:
    enabled: bool
    model_id: str
    output_dir: str
    num_images: int
    batch_size: int
    guidance_scale: float
    num_inference_steps: int
    torch_dtype: str = "fp16"
    seed: Optional[int] = None
    enable_attention_slicing: bool = True


def get_embeddings_n_labels(configs, encoder_bundle, corruption=False):
    if corruption:
        scheme = configs.scheme
    else:
        scheme = "baseline"
    # dataloader for images
    train_loader, valid_loader = dataloader.get_imagenette_loaders(
        scheme,
        configs.corrupt_range,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        shuffle=False,
    )
    train_embeddings, train_labels = extract_features(
        encoder_bundle,
        train_loader,
        scheme,
        configs.device,
        generator=configs.generator,
        noise_std=configs.noise_std,
        max_items=None,
    )
    valid_embeddings, valid_labels = extract_features(
        encoder_bundle,
        valid_loader,
        scheme,
        configs.device,
        generator=configs.generator,
        noise_std=configs.noise_std,
        max_items=None,
    )
    mix_pairs = {"train": None, "valid": None}
    mix_perm = {"train": None, "valid": None}
    if corruption and scheme == "mix":
        alpha = configs.mix_alpha if configs.mix_alpha is not None else 0.5
        gen_train = torch.Generator().manual_seed(configs.seed)
        train_embeddings, pair_train, perm_train = apply_mix_embeddings(
            train_embeddings, train_labels, alpha, gen_train
        )
        mix_pairs["train"] = pair_train
        mix_perm["train"] = perm_train

        gen_valid = torch.Generator().manual_seed(configs.seed + 1)
        valid_embeddings, pair_valid, perm_valid = apply_mix_embeddings(
            valid_embeddings, valid_labels, alpha, gen_valid
        )
        mix_pairs["valid"] = pair_valid
        mix_perm["valid"] = perm_valid
    return {
        "embeddings": {"train": train_embeddings, "valid": valid_embeddings},
        "labels": {"train": train_labels, "valid": valid_labels},
        "mix_pairs": mix_pairs,
        "mix_perm": mix_perm,
    }


def load_encoder(encoder_configs: EncoderConfigs, device: torch.device):
    encoder_type = encoder_configs.encoder_type.lower()
    model_name = encoder_configs.model_name

    if encoder_type == "dino":
        model = AutoModel.from_pretrained(model_name)
        feature_dim = getattr(model.config, "hidden_size", None)
        processor = None
    elif encoder_type == "clip":
        model_kwargs = {}
        if encoder_configs.model_subfolder:
            model_kwargs["subfolder"] = encoder_configs.model_subfolder
        model = CLIPVisionModelWithProjection.from_pretrained(
            model_name, **model_kwargs
        )

        processor_kwargs = {}
        if encoder_configs.processor_subfolder:
            processor_kwargs["subfolder"] = encoder_configs.processor_subfolder
        processor = CLIPImageProcessor.from_pretrained(
            model_name, **processor_kwargs
        )
        feature_dim = getattr(model.config, "projection_dim", None)
    else:
        raise ValueError(f"Unsupported encoder_type='{encoder_configs.encoder_type}'")

    if feature_dim is None:
        raise ValueError(
            f"Could not infer embedding dimension from model config for '{model_name}'."
        )

    model.to(device)
    model.eval()

    return {
        "model": model,
        "type": encoder_type,
        "feature_dim": feature_dim,
        "image_size": encoder_configs.image_size,
        "processor": processor,
        "normalize": encoder_configs.normalize_features,
    }


def prepare_unclip_pixel_values(
    images: torch.Tensor, processor: CLIPImageProcessor
) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    images = (images * std + mean).clamp(0.0, 1.0)
    pil_batch = [to_pil_image(img.cpu()) for img in images]
    pixel_values = processor(images=pil_batch, return_tensors="pt").pixel_values
    return pixel_values.to(images.device)


@torch.no_grad()
def extract_features(
    encoder_bundle,
    loader: DataLoader,
    scheme: str,
    device: torch.device,
    generator: torch.Generator,
    noise_std: float,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std is None and scheme.lower() == "gaussian":
        raise ValueError(f"`noise_std` must be defined for the scheme {scheme}")

    model = encoder_bundle["model"]
    encoder_type = encoder_bundle["type"]
    feature_dim = encoder_bundle["feature_dim"]
    image_size = encoder_bundle.get("image_size", 224)
    normalize_feats = encoder_bundle.get("normalize", True)
    processor = encoder_bundle.get("processor")
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    n_seen = 0
    for images, y in tqdm(loader):
        images = images.to(device, non_blocking=True)

        if encoder_type == "clip":
            if processor is None:
                raise ValueError("CLIP encoder requires an associated processor.")
            pixel_values = prepare_unclip_pixel_values(images, processor)
            outputs = model(pixel_values=pixel_values)
            z = outputs.image_embeds
        elif encoder_type == "dino":
            pixel_values = images
            outputs = model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state  # (B, N+1, C)
            hidden = hidden[:, 1:, :]  # drop CLS
            b, n, c = hidden.shape
            hw = int(math.sqrt(n))
            if hw * hw != n:
                raise ValueError(
                    f"Unexpected token count {n}, cannot reshape to square."
                )
            z = hidden.transpose(1, 2).reshape(b, c, hw, hw)
            z = z.mean(dim=(-2, -1))  # (B, C)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        if z.shape[-1] != feature_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {feature_dim}, got {z.shape[-1]}"
            )

        if normalize_feats:
            z = F.normalize(z, dim=-1)

        if scheme.lower() == "gaussian" and noise_std > 0:
            eps = torch.randn(
                z.shape, device=z.device, dtype=z.dtype, generator=generator
            )
            z = z + noise_std * eps
            if normalize_feats:
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


def apply_mix_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    generator: torch.Generator,
):
    labels = labels.clone()
    embeddings = embeddings.clone()
    unique_labels = torch.unique(labels, sorted=True)
    perm = torch.empty_like(labels, dtype=torch.long)

    class_to_indices: List[torch.Tensor] = []
    for lbl in unique_labels:
        idxs = torch.nonzero(labels == lbl, as_tuple=False).flatten()
        idxs = idxs[torch.randperm(len(idxs), generator=generator)]
        class_to_indices.append(idxs)

    for i, idxs in enumerate(class_to_indices):
        next_idxs = class_to_indices[(i + 1) % len(class_to_indices)]
        repeat = (len(idxs) + len(next_idxs) - 1) // len(next_idxs)
        expanded = next_idxs.repeat(repeat)[: len(idxs)]
        perm[idxs] = expanded

    mixed = alpha * embeddings + (1 - alpha) * embeddings[perm]
    pair_labels = torch.stack([labels, labels[perm]], dim=1)
    return mixed, pair_labels, perm


def _select_unique_mix_pairs(
    mix_perm: Optional[torch.Tensor],
    mix_pairs: Optional[torch.Tensor],
    max_pairs: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if mix_perm is None:
        return None

    perm = mix_perm.view(-1).cpu()
    pair_labels = mix_pairs.cpu() if mix_pairs is not None else None

    idx_a: List[int] = []
    idx_b: List[int] = []
    labels: List[torch.Tensor] = []
    seen = set()

    for i in range(len(perm)):
        partner = int(perm[i])
        if partner == i:
            continue

        key = tuple(sorted((i, partner)))
        if key in seen:
            continue

        if pair_labels is not None:
            lbl_pair = pair_labels[i]
            if lbl_pair[0].item() == lbl_pair[1].item():
                continue
            labels.append(lbl_pair)

        idx_a.append(i)
        idx_b.append(partner)
        seen.add(key)

        if max_pairs is not None and len(idx_a) >= max_pairs:
            break

    if not idx_a:
        return None

    idx_a_tensor = torch.tensor(idx_a, dtype=torch.long)
    idx_b_tensor = torch.tensor(idx_b, dtype=torch.long)
    label_tensor = torch.stack(labels, dim=0) if labels else None

    return idx_a_tensor, idx_b_tensor, label_tensor


def build_mixed_embedding_sets(
    reconstructed_embeddings: torch.Tensor,
    raw_embeddings: torch.Tensor,
    mix_perm: Optional[torch.Tensor],
    mix_pairs: Optional[torch.Tensor],
    alpha: Optional[float],
    normalize: bool,
    max_pairs: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    pair_selection = _select_unique_mix_pairs(mix_perm, mix_pairs, max_pairs)
    if pair_selection is None:
        print("No suitable mix pairs found for reconstructed embedding mixing.")
        return None

    idx_a, idx_b, label_pairs = pair_selection
    mix_alpha = alpha if alpha is not None else 0.5

    def _mix(embeds: torch.Tensor) -> torch.Tensor:
        mixed = mix_alpha * embeds[idx_a] + (1.0 - mix_alpha) * embeds[idx_b]
        if normalize:
            mixed = F.normalize(mixed, dim=-1)
        return mixed

    recon_mix = _mix(reconstructed_embeddings)
    raw_mix = _mix(raw_embeddings)

    metadata: Dict[str, torch.Tensor] = {
        "mix_reconstructed": recon_mix,
        "mix_raw": raw_mix,
        "pair_indices": torch.stack((idx_a, idx_b), dim=1),
    }
    if label_pairs is not None:
        metadata["pair_labels"] = label_pairs

    print(
        f"Prepared {recon_mix.size(0)} mixed embeddings (alpha={mix_alpha:.2f})"
        " for reconstructed vs. raw baselines."
    )
    return metadata


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


def reconstruct(
    global_configs: GlobalConfigs,
    encoder_configs: EncoderConfigs,
    restore_configs: ReconstructConfigs,
    unclip_configs: Optional[UnCLIPConfigs] = None,
):
    encoder_bundle = load_encoder(encoder_configs, global_configs.device)
    normalize_features = encoder_bundle.get("normalize", True)
    clean_embeds_n_labels = get_embeddings_n_labels(global_configs, encoder_bundle)
    corrupt_embeds_n_labels = get_embeddings_n_labels(
        global_configs, encoder_bundle, corruption=True
    )

    clean_train_embeddings = clean_embeds_n_labels["embeddings"]["train"]
    corrupt_train_embeddings = corrupt_embeds_n_labels["embeddings"]["train"]
    clean_valid_embeddings = clean_embeds_n_labels["embeddings"]["valid"]
    corrupt_valid_embeddings = corrupt_embeds_n_labels["embeddings"]["valid"]
    train_labels = clean_embeds_n_labels["labels"]["train"]
    valid_labels = clean_embeds_n_labels["labels"]["valid"]
    mix_pairs_valid = corrupt_embeds_n_labels["mix_pairs"]["valid"]
    mix_perm_valid = corrupt_embeds_n_labels["mix_perm"]["valid"]

    if restore_configs.method == "ridge":
        W, b = fit_ridge_linear_map(
            X_train=corrupt_train_embeddings,
            Y_train=clean_train_embeddings,
            lam=restore_configs.lam,
        )
        train_embeds_pred = linear_map_predict(corrupt_train_embeddings, W, b)
        valid_embeds_pred = linear_map_predict(corrupt_valid_embeddings, W, b)

    if restore_configs.method in ["mlp", "hopfield", "hopfield-associate"]:
        # dataloader for image embeddings
        train_loader, train_eval_loader, val_loader = create_dataloaders(
            global_configs,
            restore_configs,
            clean_train_embeddings,
            corrupt_train_embeddings,
            clean_valid_embeddings,
            corrupt_valid_embeddings,
        )
        feature_dim = encoder_bundle["feature_dim"]
        model, losses = train(
            restore_configs,
            train_loader,
            val_loader,
            global_configs.device,
            feature_dim,
            normalize_features,
        )
        model.eval()
        train_embeds_pred, valid_embeds_pred = evaluate(
            model,
            train_eval_loader,
            val_loader,
            global_configs.device,
            normalize_features,
            restore_method=restore_configs.method,
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

    mix_metadata = None
    if (
        global_configs.scheme == "mix"
        and mix_perm_valid is not None
        and clean_valid_embeddings.size(0) > 0
        and unclip_configs is not None
        and unclip_configs.enabled
    ):
        max_pairs = unclip_configs.num_images
        mix_metadata = build_mixed_embedding_sets(
            reconstructed_embeddings=valid_embeds_pred,
            raw_embeddings=clean_valid_embeddings,
            mix_perm=mix_perm_valid,
            mix_pairs=mix_pairs_valid,
            alpha=global_configs.mix_alpha,
            normalize=normalize_features,
            max_pairs=max_pairs,
        )

    if encoder_configs.encoder_type.lower() == "clip":
        pipeline = None
        seed_offset = 0
        pipeline, seed_offset = maybe_generate_images_with_unclip(
            valid_embeds_pred,
            unclip_configs,
            global_configs.device,
        )
        if mix_metadata is not None:
            pipeline, seed_offset = maybe_generate_images_with_unclip(
                mix_metadata.get("mix_reconstructed"),
                unclip_configs,
                global_configs.device,
                output_dir_suffix="mix_reconstructed",
                filename_prefix="mix_recon",
                seed_offset=seed_offset,
                pipeline=pipeline,
            )
            pipeline, seed_offset = maybe_generate_images_with_unclip(
                mix_metadata.get("mix_raw"),
                unclip_configs,
                global_configs.device,
                output_dir_suffix="mix_raw",
                filename_prefix="mix_raw",
                seed_offset=seed_offset,
                pipeline=pipeline,
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
    train_eval_loader = DataLoader(
        train_ds, batch_size=global_configs.batch_size, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=global_configs.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, train_eval_loader, val_loader


def train(
    restore_configs,
    train_loader,
    val_loader,
    device,
    feature_dim: int,
    normalize_outputs: bool,
):
    in_dim = feature_dim
    out_dim = feature_dim
    restore_method = restore_configs.method

    if restore_method == "mlp":
        model = ResidualMLP(
            in_dim,
            restore_configs.hidden_dims,
            out_dim,
            num_layers=restore_configs.num_layers,
            dropout=restore_configs.dropout,
        ).to(device)
    if restore_method == "hopfield":
        model = HopfieldLayerDecoder(embeds_dim=in_dim).to(device)

    if restore_method == "hopfield-associate":
        model = HopfieldAssociateDecoder(embeds_dim=in_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    num_epochs = restore_configs.epochs
    losses = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if restore_method in ["mlp", "hopfield"]:
                y_pred = model(xb)
            if restore_method == "hopfield-associate":
                y_pred = model(xb, yb)

            if normalize_outputs:
                y_pred = F.normalize(y_pred, dim=-1)
            loss = F.mse_loss(y_pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        epoch_train_loss = running_loss / max(n_samples, 1)
        losses["train"].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                if restore_method in ["mlp", "hopfield"]:
                    preds = model(xb)
                if restore_method == "hopfield-associate":
                    preds = model(xb, yb)

                if normalize_outputs:
                    preds = F.normalize(preds, dim=-1)
                batch_loss = F.mse_loss(preds, yb)
                val_loss += batch_loss.item() * xb.size(0)
                val_samples += xb.size(0)
        epoch_val_loss = val_loss / max(val_samples, 1)
        losses["val"].append(epoch_val_loss)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - train_loss: {epoch_train_loss:.6f} - val_loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            best_state = model.state_dict()
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    losses["best_epoch"] = best_epoch
    losses["best_val_loss"] = best_val

    return model, losses


def evaluate(
    model,
    train_eval_loader,
    val_loader,
    device,
    normalize_outputs: bool,
    restore_method: str = "mlp",
):
    model.eval()
    yhat_train = []
    with torch.no_grad():
        for xb, yb in train_eval_loader:
            xb = xb.to(device)
            if restore_method == "hopfield-associate":
                yb = yb.to(device)
                preds = model(xb, yb)
            else:
                preds = model(xb)
            if normalize_outputs:
                preds = F.normalize(preds, dim=-1)
            yhat_train.append(preds.cpu())
    train_pred = torch.cat(yhat_train, dim=0)

    yhat_val = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            if restore_method == "hopfield-associate":
                yb = yb.to(device)
                preds = model(xb, yb)
            else:
                preds = model(xb)
            if normalize_outputs:
                preds = F.normalize(preds, dim=-1)
            yhat_val.append(preds.cpu())
    val_pred = torch.cat(yhat_val, dim=0)

    return train_pred, val_pred


def maybe_generate_images_with_unclip(
    embeddings: Optional[torch.Tensor],
    unclip_configs: Optional[UnCLIPConfigs],
    device: torch.device,
    output_dir_suffix: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    seed_offset: int = 0,
    pipeline: Optional[StableUnCLIPImg2ImgPipeline] = None,
) -> Tuple[Optional[StableUnCLIPImg2ImgPipeline], int]:
    if (
        embeddings is None
        or embeddings.numel() == 0
        or unclip_configs is None
        or not unclip_configs.enabled
    ):
        return pipeline, seed_offset

    torch_device = torch.device(device)
    num_embeddings = embeddings.size(0)
    num_to_generate = min(unclip_configs.num_images, num_embeddings)
    if num_to_generate <= 0:
        print("Stable unCLIP generation skipped (num_images <= 0).")
        return pipeline, seed_offset

    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(unclip_configs.torch_dtype.lower(), torch.float32)

    target_dir = (
        os.path.join(unclip_configs.output_dir, output_dir_suffix)
        if output_dir_suffix
        else unclip_configs.output_dir
    )
    os.makedirs(target_dir, exist_ok=True)
    prefix = filename_prefix or (output_dir_suffix or "valid")

    if pipeline is None:
        print(
            f"Loading Stable unCLIP pipeline '{unclip_configs.model_id}' "
            f"for decoding up to {num_to_generate} embeddings..."
        )
        pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            unclip_configs.model_id,
            torch_dtype=target_dtype,
        )
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=True)
        if unclip_configs.enable_attention_slicing:
            pipeline.enable_attention_slicing()

    selected = embeddings[:num_to_generate].to(device=torch_device, dtype=target_dtype)
    batch_size = max(1, unclip_configs.batch_size)
    total_generated = 0

    for start in range(0, num_to_generate, batch_size):
        end = min(start + batch_size, num_to_generate)
        batch = selected[start:end]
        generator = None
        if unclip_configs.seed is not None:
            generator = torch.Generator(device=torch_device)
            generator.manual_seed(unclip_configs.seed + seed_offset + start)

        prompts = [""] * (end - start)
        outputs = pipeline(
            image=None,
            prompt=prompts,
            image_embeds=batch,
            guidance_scale=unclip_configs.guidance_scale,
            num_inference_steps=unclip_configs.num_inference_steps,
            generator=generator,
        )
        for idx, pil_img in enumerate(outputs.images):
            global_idx = start + idx
            file_path = os.path.join(target_dir, f"{prefix}_{global_idx:04d}.png")
            pil_img.save(file_path)
            print(f"Saved unCLIP image -> {file_path}")

        total_generated += end - start

    return pipeline, seed_offset + total_generated


def main():
    (
        global_configs,
        encoder_configs,
        restore_configs,
        unclip_configs,
    ) = set_up()
    reconstruct(global_configs, encoder_configs, restore_configs, unclip_configs)


if __name__ == "__main__":
    main()
