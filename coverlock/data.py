"""Natural-image pairs and the distortion distribution used for training."""

import io
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class RandomDistortion:
    names = ("identity", "jpeg", "gaussian_blur", "gaussian_noise",
             "crop", "brightness", "contrast")

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = TF.resize(image.convert("RGB"), [self.size, self.size], antialias=True)
        attack = random.choice(self.names)
        if attack == "identity":
            return TF.to_tensor(image)
        if attack == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=random.randint(45, 90), subsampling=2)
            return TF.to_tensor(Image.open(buffer).convert("RGB"))
        if attack == "gaussian_blur":
            return TF.to_tensor(image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0))))
        tensor = TF.to_tensor(image)
        if attack == "gaussian_noise":
            return (tensor + torch.randn_like(tensor) * random.uniform(0.005, 0.02)).clamp(0, 1)
        if attack == "crop":
            side = max(1, int(self.size * random.uniform(0.70, 0.95) ** 0.5))
            top = random.randint(0, self.size - side)
            left = random.randint(0, self.size - side)
            mask = torch.zeros_like(tensor)
            mask[:, top:top + side, left:left + side] = 1
            return tensor * mask
        if attack == "brightness":
            return TF.to_tensor(ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3)))
        return TF.to_tensor(ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3)))


class ImagePairs(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root, size=224, seed=0):
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in self.extensions)
        if len(paths) < 2:
            raise ValueError(f"need at least two images under {root}")
        random.Random(seed).shuffle(paths)
        # Match the final training runs: reserve the last 20% for evaluation.
        self.paths = paths[:max(1, int(len(paths) * 0.8))]
        self.size = size
        self.distort = RandomDistortion(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        clean = TF.to_tensor(TF.resize(image, [self.size, self.size], antialias=True))
        return clean, self.distort(image)
