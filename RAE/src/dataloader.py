from corruption import RandomAreaDownUp, RandomSquareOccluder
from fastai.vision.all import untar_data, URLs
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def preprocess(scheme: str, corrupt_range: tuple):
    # standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Use deterministic resizing for both splits so corresponding clean/corrupt
    # samples stay aligned across repeated feature extractions.
    base_tfms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    normalize = transforms.Normalize(mean, std)
    if scheme in ["baseline", "gaussian"]:
        corrupt_tfm = []
    elif scheme == "crop":
        corrupt_tfm = [RandomAreaDownUp(out_size=224, area_frac_range=corrupt_range)]
    elif scheme == "occlude":
        corrupt_tfm = [RandomSquareOccluder(area_frac_range=corrupt_range, fill=0.0)]

    train_tfms = transforms.Compose(base_tfms + corrupt_tfm + [normalize])
    valid_tfms = transforms.Compose(base_tfms + corrupt_tfm + [normalize])

    return train_tfms, valid_tfms


def get_imagenette_loaders(
    scheme: str,
    corrupt_range: tuple,
    size: str = "320px",  # "full", "320px", or "160px"
    batch_size: int = 64,
    num_workers: int = 1,
    shuffle: bool = True,
):
    if corrupt_range is None and scheme not in ["baseline", "gaussian"]:
        raise ValueError(f"`corrupt_range` must be provided for scheme='{scheme}'. ")

    # Map human-readable size to fastai URLs
    url_map = {
        "full": URLs.IMAGENETTE,
        "320px": URLs.IMAGENETTE_320,
        "160px": URLs.IMAGENETTE_160,
    }
    if size not in url_map:
        raise ValueError(f"size must be one of {list(url_map.keys())}, got {size!r}")

    # This will *reuse* the already-downloaded dataset if it exists
    root = Path(untar_data(url_map[size]))  # e.g. .../imagenette2-160
    train_dir = root / "train"
    val_dir = root / "val"

    train_tfms, val_tfms = preprocess(scheme, corrupt_range)

    # Imagenette is laid out as train/cls_name/*.jpg, val/cls_name/*.jpg
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    scheme = "baseline"
    corrupt_range = None
    train_loader, val_loader = get_imagenette_loaders(
        scheme, corrupt_range, batch_size=1
    )
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    for xb, yb in train_loader:
        # xb: (B, 3, 224, 224), yb: (B,)
        print(xb.shape, yb)
        unnorm = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std],
        )
        img_unnorm = unnorm(xb[0, :, :, :]).clamp(0, 1)
        pil_img = transforms.functional.to_pil_image(img_unnorm)
        pil_img.save("sample_train_image.jpg")
        break
