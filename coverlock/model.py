"""Inference model for CoverLock checkpoints."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF


BACKBONES = {
    "dinov2_vits14": (384, 14),
    "dinov2_vitb14": (768, 14),
    "dinov2_vitl14": (1024, 14),
    "dinov2_vitg14": (1536, 14),
}


def load_backbone(name: str) -> tuple[nn.Module, int, int]:
    if name not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}")
    cache = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if cache.exists():
        model = torch.hub.load(str(cache), name, pretrained=True, source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    return model, *BACKBONES[name]


class FrozenDINOExtractor(nn.Module):
    def __init__(self, backbone: str, layer: int):
        super().__init__()
        self.backbone, self.output_dim, self.patch_size = load_backbone(backbone)
        self.layer = layer
        self.backbone.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        normalized = (images - mean) / std
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                normalized, n=[self.layer], reshape=False,
                return_class_token=True, norm=True,
            )
        tokens = features[0][0] if isinstance(features[0], tuple) else features[0]
        height = normalized.shape[-2] // self.patch_size
        width = normalized.shape[-1] // self.patch_size
        tokens = F.normalize(tokens, dim=-1)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)


class CoverLockHead(nn.Module):
    """Mean/std pooling, MLP embedding, and fixed keyed projection."""

    def __init__(self, feature_channels: int, message_bits: int = 256,
                 hidden: int = 512, embedding_dim: int = 512,
                 hash_seed: int = 0, projection_mode: str = "orthogonal"):
        super().__init__()
        if projection_mode not in {"gaussian_normalized", "orthogonal"}:
            raise ValueError("invalid projection mode")
        if projection_mode == "orthogonal" and message_bits > embedding_dim:
            raise ValueError("orthogonal projection requires bits <= embedding dimension")
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_channels * 2),
            nn.Linear(feature_channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(hash_seed))
        projection = torch.randn(embedding_dim, message_bits, generator=generator)
        if projection_mode == "orthogonal":
            projection = torch.linalg.qr(projection, mode="reduced").Q
        else:
            projection = F.normalize(projection, dim=0)
        self.register_buffer("hash_projection", projection)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=(-2, -1))
        std = features.std(dim=(-2, -1), unbiased=False)
        return F.normalize(self.encoder(torch.cat([mean, std], dim=1)), dim=1)

    def forward(self, features: torch.Tensor, return_embeddings: bool = False):
        embeddings = self.embeddings(features)
        logits = embeddings @ self.hash_projection * self.logit_scale.exp().clamp(max=100.0)
        return (logits, embeddings) if return_embeddings else logits


class CoverLock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.image_size = int(config["image_size"])
        self.extractor = FrozenDINOExtractor(config["backbone"], int(config["layer"]))
        self.head = CoverLockHead(
            self.extractor.output_dim,
            int(config["message_bits"]), int(config["hidden"]),
            int(config["embedding_dim"]), int(config["hash_seed"]),
            config["projection_mode"],
        )

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["config"])
        model.head.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(device).eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.extractor(images))

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self(images).ge(0)

    @torch.inference_mode()
    def encode_pil(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image.convert("RGB"), [self.image_size, self.image_size], antialias=True)
        tensor = TF.to_tensor(image).unsqueeze(0).to(next(self.parameters()).device)
        return self.encode(tensor)
