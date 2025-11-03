import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor

try:
    import webdataset as wds
except ModuleNotFoundError:
    wds = None  # noqa: N816

from stage1.encoders.dinov2 import Dinov2withNorm
from .model import FeatureMaskedAutoencoder
from .utils import FeatureNormalizer, build_rae_decoder, save_reconstruction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINO-feature Masked Autoencoder.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet data or WebDataset shards.")
    parser.add_argument("--dinov2-path", type=str, default="models/hf/facebook__dinov2-with-registers-base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=1.5e-4, help="Base learning rate for a batch_size=256.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/feature_mae")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=20, help="Save checkpoint every N epochs.")
    parser.add_argument("--visualize-interval", type=int, default=1000, help="Steps between recon visualisations.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per worker (if >0 workers).")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16", help="AMP compute precision.")
    parser.add_argument("--feature-stat-path", type=str, default="models/stats/dinov2/wReg_base/imagenet1k/stat.pt")
    parser.add_argument("--decoder-config-path", type=str, default="configs/decoder/ViTXL")
    parser.add_argument("--decoder-weights", type=str, default="models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt")
    parser.add_argument("--save-images", action="store_true", help="Enable saving reconstructions.")
    parser.add_argument("--include-special", action="store_true", help="Include CLS + register tokens as inputs/targets.")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Number of micro-batches per epoch for iterable datasets (WebDataset).",
    )
    parser.add_argument(
        "--steps-are-global",
        action="store_true",
        help="Interpret steps-per-epoch as global across all ranks; per-rank steps are computed as ceil(steps/world_size).",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training from.")
    parser.add_argument("--compile", dest="use_compile", action="store_true", help="(Deprecated) no-op flag retained for compatibility.")
    parser.add_argument("--no-compile", dest="use_compile", action="store_false")

    parser.add_argument("--encoder-embed-dim", type=int, default=768)
    parser.add_argument("--encoder-depth", type=int, default=12)
    parser.add_argument("--encoder-heads", type=int, default=12)
    parser.add_argument("--decoder-embed-dim", type=int, default=512)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--decoder-heads", type=int, default=16)

    parser.set_defaults(amp=True, use_compile=False)
    return parser.parse_args()


def init_distributed_mode(args: argparse.Namespace) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0


def is_master(args: argparse.Namespace) -> bool:
    return args.rank == 0


def set_random_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_transform(
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    input_is_tensor: bool = False,
    apply_normalize: bool = True,
) -> transforms.Compose:
    ops = [
        transforms.RandomResizedCrop(
            image_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
    ]
    # When decoding to PIL, add ToTensor; for torchrgb, tensors are already float in [0,1]
    if not input_is_tensor:
        ops.append(transforms.ToTensor())
    if apply_normalize:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def build_dataloader(
    args: argparse.Namespace,
    transform: transforms.Compose,
) -> Tuple[Iterable, Optional[DistributedSampler]]:
    root = Path(args.data_path)
    tar_files = sorted(root.rglob("*.tar"))
    if tar_files:
        if wds is None:
            raise ImportError(
                "webdataset is required to load *.tar shards. Install it with `pip install webdataset` inside the rae environment."
            )
        shard_pattern = [str(p) for p in tar_files]
        dataset = (
            wds.WebDataset(
                shard_pattern,
                repeat=True,
                handler=wds.handlers.warn_and_continue,
                shardshuffle=200,
                # Explicitly split shards across distributed ranks
                nodesplitter=getattr(wds, "split_by_node", None),
            )
            # Larger sample-level shuffle buffer improves randomness for large batches
            .shuffle(10000, initial=10000)
            .decode("torchrgb")
            .to_tuple("jpg;jpeg;png", "cls")
            .map_tuple(transform, lambda x: x)
            .batched(args.batch_size, partial=False)
        )
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        )
        if args.world_size > 1 and hasattr(loader, "ddp_equalize"):
            loader = loader.ddp_equalize(args.world_size)
        return loader, None

    dataset = ImageFolder(str(root), transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True) if args.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    return loader, sampler


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    progress: float,
    *,
    base_lr: float,
    min_lr: float,
    warmup: float,
) -> float:
    if progress < warmup:
        lr = base_lr * progress / max(warmup, 1e-8)
    else:
        cosine_progress = (progress - warmup) / max(1.0 - warmup, 1e-8)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def _extract_images(batch) -> torch.Tensor:
    # Handles outputs from ImageFolder DataLoader and WebDataset (with/without dataset-side batching)
    if isinstance(batch, (list, tuple)):
        first = batch[0]
        # Case: dataset-side batching -> first is a list of tensors
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], torch.Tensor):
            return torch.stack(first, dim=0)
        # Case: regular tuple (tensor, label)
        if isinstance(first, torch.Tensor):
            return first
        return first
    if isinstance(batch, dict):
        return batch["jpg"] if "jpg" in batch else batch["png"]
    return batch


def train_one_epoch(
    model: nn.Module,
    dino: nn.Module,
    loader: Iterable,
    sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    normalizer: Optional[FeatureNormalizer],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    decoder: Optional[nn.Module],
    epoch: int,
    args: argparse.Namespace,
) -> float:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    total_loss = torch.zeros(1, device=device)
    try:
        loader_length = len(loader)
    except TypeError:
        loader_length = None
    total_steps = args.steps_per_epoch if args.steps_per_epoch is not None else loader_length
    # For iterable datasets (e.g., WebDataset), optionally treat steps_per_epoch as global
    if loader_length is None and total_steps is not None and total_steps > 0:
        if args.world_size > 1 and getattr(args, "steps_are_global", False):
            import math as _math
            per_rank = _math.ceil(total_steps / args.world_size)
            if is_master(args):
                print(
                    f"Adjusting steps_per_epoch from global {total_steps} to per-rank {per_rank} for world_size={args.world_size}",
                    flush=True,
                )
            total_steps = per_rank
    if total_steps is None or total_steps <= 0:
        raise ValueError("steps_per_epoch must be provided for iterable datasets.")

    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(args.accumulation_steps, 1)
    grad_acc_step = 0

    start_time = time.time()
    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        images = _extract_images(batch).to(device, non_blocking=True)
        # Optimize memory layout for H100 tensor cores
        if images.is_floating_point():
            images = images.to(memory_format=torch.channels_last)
        # Normalize on device for WebDataset to offload CPU
        if getattr(args, "normalize_on_device", False):
            images = (images - image_mean.view(1, -1, 1, 1)) / image_std.view(1, -1, 1, 1)

        autocast_ctx = autocast(**args.autocast_kwargs) if args.autocast_kwargs else nullcontext()
        with torch.no_grad():
            # Run DINO feature extraction in AMP as well for speed
            with autocast_ctx:
                features = dino(images)
                if normalizer is not None:
                    features = normalizer.normalize(features)

        with autocast_ctx:
            pred, target, mask = model(features, mask_ratio=args.mask_ratio)
            loss_fn = model.module.loss if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.loss
            loss = loss_fn(pred, target, mask) / accum_steps

        scaler.scale(loss).backward()
        grad_acc_step += 1
        if grad_acc_step == accum_steps:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            grad_acc_step = 0

        total_loss += loss.detach() * accum_steps

        global_step = epoch * total_steps + step + 1
        lr = adjust_learning_rate(
            optimizer,
            progress=global_step / (args.epochs * total_steps),
            base_lr=getattr(args, "effective_base_lr", args.base_lr),
            min_lr=getattr(args, "effective_min_lr", args.min_lr),
            warmup=args.warmup_epochs / args.epochs,
        )

        if is_master(args) and (step % args.log_interval == 0 or step + 1 == total_steps):
            current_loss = loss.detach() * accum_steps
            print(
                f"Epoch {epoch} Step {step+1}/{total_steps} "
                f"Loss {current_loss.item():.4f} LR {lr:.6f} "
                f"Time {(time.time() - start_time):.2f}s",
                flush=True,
            )

        if (
            decoder is not None
            and args.save_images
            and is_master(args)
            and (step + 1) % max(args.visualize_interval, 1) == 0
        ):
            with torch.no_grad():
                recon_tokens = target * (1.0 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
                if normalizer is not None:
                    recon_tokens = normalizer.denormalize(recon_tokens)

                # For visualization, drop special tokens if present and only pass patch tokens
                if args.include_special:
                    start = dino.num_special_tokens
                    recon_tokens_vis = recon_tokens[:, start:, :]
                else:
                    recon_tokens_vis = recon_tokens
                cls = torch.zeros(recon_tokens_vis.size(0), 1, recon_tokens_vis.size(-1), device=recon_tokens_vis.device)
                decoder_input = torch.cat([cls, recon_tokens_vis], dim=1)
                logits = decoder(decoder_input, drop_cls_token=False).logits
                pixels = decoder.unpatchify(logits)
                save_path = Path(args.output_dir) / "samples" / f"epoch{epoch:03d}_step{step:06d}.png"
                save_reconstruction_grid(pixels.cpu(), mean=image_mean.cpu(), std=image_std.cpu(), filename=save_path)

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= args.world_size
    return (total_loss.item() / total_steps) if total_steps > 0 else total_loss.item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    args: argparse.Namespace,
    best_loss: Optional[float] = None,
) -> None:
    if not is_master(args):
        return
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    if best_loss is not None:
        state["best_loss"] = best_loss
    path = ckpt_dir / f"checkpoint_{epoch:03d}.pth"
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def main() -> None:
    args = parse_args()
    init_distributed_mode(args)

    args.precision = args.precision.lower()
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if args.precision == "bf16":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if not bf16_supported:
                if is_master(args):
                    print("BF16 not supported on this GPU; falling back to FP16.")
                args.precision = "fp16"
    else:
        if args.precision != "fp32" and is_master(args):
            print("CUDA not available; running in FP32 without AMP.")
        args.precision = "fp32"
        args.amp = False

    if args.precision == "fp32":
        args.amp = False

    if args.amp and torch.cuda.is_available():
        autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        args.autocast_kwargs = {"enabled": True, "dtype": autocast_dtype}
    else:
        args.autocast_kwargs = None

    set_random_seed(args.seed, args.rank)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    resume_state = None
    start_epoch = 0
    best_loss = float("inf")
    resume_path = Path(args.resume).expanduser() if args.resume else None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found at {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        best_loss = resume_state.get("best_loss", float("inf"))
        start_epoch = resume_state.get("epoch", -1) + 1
        if is_master(args):
            print(f"Resuming from {resume_path} at epoch {start_epoch}")

    try:
        processor = AutoImageProcessor.from_pretrained(args.dinov2_path, use_fast=True)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(args.dinov2_path)
    image_mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(-1)
    image_std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(-1)
    data_root = Path(args.data_path)
    has_tars = any(data_root.rglob("*.tar"))
    # For WebDataset (tensor decode), defer Normalize to GPU to reduce CPU overhead
    args.normalize_on_device = bool(has_tars)
    transform = create_transform(
        args.image_size,
        image_mean.cpu(),
        image_std.cpu(),
        input_is_tensor=has_tars,
        apply_normalize=not args.normalize_on_device,
    )

    loader, sampler = build_dataloader(args, transform)

    dino = Dinov2withNorm(dinov2_path=args.dinov2_path, include_special_tokens=args.include_special)
    dino.to(device)
    dino.eval()
    dino.requires_grad_(False)
    feature_dim = dino.hidden_size
    num_patches = (args.image_size // dino.patch_size) ** 2
    num_tokens = num_patches + (dino.num_special_tokens if args.include_special else 0)

    model = FeatureMaskedAutoencoder(
        feature_dim=feature_dim,
        num_tokens=num_tokens,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_heads,
    )
    if resume_state is not None:
        incompatible = model.load_state_dict(resume_state["model"], strict=False)
        if is_master(args):
            missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
            if missing:
                print(f"Warning: missing keys when loading checkpoint: {missing}")
            if unexpected:
                print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
    model.to(device)

    if args.use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as exc:
            if is_master(args):
                print(f"torch.compile failed ({exc}); continuing without compilation.")
            args.use_compile = False

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False
        )

    eff_batch = args.batch_size * max(args.world_size, 1) * max(args.accumulation_steps, 1)
    scaled_base_lr = args.base_lr * eff_batch / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_base_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    args.effective_base_lr = scaled_base_lr
    args.effective_min_lr = args.min_lr * eff_batch / 256
    scaler = GradScaler(enabled=args.amp and args.precision == "fp16" and torch.cuda.is_available())

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler_state = resume_state.get("scaler", None)
        if scaler_state is not None and args.amp and args.precision == "fp16" and torch.cuda.is_available():
            scaler.load_state_dict(scaler_state)
        resume_state = None

    # FeatureNormalizer assumes a square grid (patch tokens only). Disable when special tokens are included.
    if args.include_special:
        normalizer = None
    else:
        normalizer = FeatureNormalizer(args.feature_stat_path).to(device) if args.feature_stat_path else None

    decoder = None
    if args.save_images:
        decoder = build_rae_decoder(
            args.decoder_config_path,
            args.decoder_weights,
            hidden_size=feature_dim,
            num_patches=num_patches,
            device=device,
        )

    if start_epoch >= args.epochs:
        if is_master(args):
            print(f"Start epoch {start_epoch} >= total epochs {args.epochs}; nothing to resume.")
        return

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model,
            dino,
            loader,
            sampler,
            optimizer,
            scaler,
            normalizer,
            image_mean,
            image_std,
            decoder,
            epoch,
            args,
        )

        if is_master(args):
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        if avg_loss < best_loss and is_master(args):
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scaler, epoch, args, best_loss=best_loss)
        elif args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, args, best_loss=best_loss)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
