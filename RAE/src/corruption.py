from PIL import Image
from torchvision import transforms
from typing import Tuple, Union
import math
import random
import torch
import torchvision.transforms.functional as F


class RandomAreaDownUp:
    """
    Randomly crop a region that is 20–40% of the original area, then resize back to out_size.
    Equivalent to a strong RandomResizedCrop used *as corruption*.
    Works with PIL.Image or torch.Tensor (C,H,W) in [0,1].
    """

    def __init__(
        self,
        out_size: Union[int, Tuple[int, int]] = 224,
        area_frac_range: Tuple[float, float] = (0.20, 0.40),
        ratio_range: Tuple[float, float] = (3 / 4, 4 / 3),  # allow some aspect jitter
        interpolation=F.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        self.out_size = out_size
        self.area_frac_range = area_frac_range
        self.ratio_range = ratio_range
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_size(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            # tensor C,H,W
            h, w = img.shape[-2], img.shape[-1]
        return h, w

    def __call__(self, img):
        h, w = self._get_size(img)
        area = h * w

        # Sample target area and aspect ratio
        for _ in range(10):  # try a few times to get a valid crop
            target_frac = random.uniform(*self.area_frac_range)
            target_area = target_frac * area
            log_ratio = (math.log(self.ratio_range[0]), math.log(self.ratio_range[1]))
            aspect = math.exp(random.uniform(*log_ratio))

            hh = int(round(math.sqrt(target_area * aspect)))
            ww = int(round(math.sqrt(target_area / aspect)))

            if 1 <= hh <= h and 1 <= ww <= w:
                top = random.randint(0, h - hh)
                left = random.randint(0, w - ww)

                if isinstance(img, Image.Image):
                    cropped = F.crop(img, top, left, hh, ww)
                    return F.resize(
                        cropped,
                        self.out_size,
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    )
                else:
                    # tensor path
                    cropped = img[..., top : top + hh, left : left + ww]
                    return F.resize(
                        cropped,
                        self.out_size,
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    )

        # Fallback: just resize (no corruption) if we failed to sample
        if isinstance(img, Image.Image):
            return F.resize(
                img,
                self.out_size,
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        return F.resize(
            img,
            self.out_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )


class RandomSquareOccluder:
    """
    Adds a square occluder covering 25–50% of the image *area* at a random location.
    Works with PIL.Image or torch.Tensor (C,H,W). Fill can be scalar or per-channel tuple/list.
    Recommended to apply on *tensors in [0,1]* before normalization.
    """

    def __init__(
        self,
        area_frac_range: Tuple[float, float] = (0.25, 0.50),
        fill: Union[float, Tuple[float, float, float]] = 0.0,  # black by default
    ):
        self.area_frac_range = area_frac_range
        self.fill = fill

    def _ensure_tensor(self, img):
        if isinstance(img, Image.Image):
            return F.to_tensor(img), True  # (C,H,W), flag says we converted
        return img, False

    def __call__(self, img):
        x, was_pil = self._ensure_tensor(img)  # x is tensor (C,H,W) in [0,1]
        c, h, w = x.shape
        area = h * w

        frac = random.uniform(*self.area_frac_range)
        side = int(round(math.sqrt(frac * area)))  # occluder side length (pixels)
        side = max(1, min(side, h, w))

        top = random.randint(0, h - side)
        left = random.randint(0, w - side)

        if isinstance(self.fill, (tuple, list)):
            # per-channel fill
            fill_tensor = torch.tensor(self.fill, dtype=x.dtype, device=x.device).view(
                c, 1, 1
            )
        else:
            fill_tensor = torch.tensor(
                float(self.fill), dtype=x.dtype, device=x.device
            ).view(1, 1, 1)

        x[..., top : top + side, left : left + side] = fill_tensor

        if was_pil:
            return F.to_pil_image(x.clamp(0, 1))
        return x.clamp(0, 1)
