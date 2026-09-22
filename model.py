"""Architecture NB6 et chargement du checkpoint Faster R-CNN.

Le ViT reprend les noms de modules du dépôt officiel Meta I-JEPA afin que les
clés du ``state_dict`` généré dans ``notebook6(2).ipynb`` correspondent
exactement. Aucun accès réseau n'est nécessaire au lancement de l'application.
"""

from __future__ import annotations

import math
import gc
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

from config import (
    CLASS_NAMES,
    EMBED_DIM,
    IMAGE_MEAN,
    IMAGE_STD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PATCH_SIZE,
)


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("La dimension du positional embedding doit être paire.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    values = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(values), np.cos(values)], axis=1)


def _sincos_2d(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x)), attention


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """ViT-S/16 compatible avec ``src.models.vision_transformer.vit_small``."""

    def __init__(self, rectangular_pos_embed: bool = False) -> None:
        super().__init__()
        self.num_features = self.embed_dim = EMBED_DIM
        self.num_heads = 6
        self.patch_embed = PatchEmbed(224, PATCH_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, EMBED_DIM),
            requires_grad=False,
        )
        positional = _sincos_2d(EMBED_DIM, 14)
        self.pos_embed.data.copy_(
            torch.from_numpy(positional).float().unsqueeze(0)
        )
        if rectangular_pos_embed:
            rectangular = interpolate_positional_embedding(
                self.pos_embed.detach(),
                INPUT_HEIGHT // PATCH_SIZE,
                INPUT_WIDTH // PATCH_SIZE,
            )
            self.pos_embed = nn.Parameter(rectangular, requires_grad=False)
            self.patch_embed.num_patches = (
                INPUT_HEIGHT // PATCH_SIZE
            ) * (INPUT_WIDTH // PATCH_SIZE)
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(
                "Nombre de tokens incompatible avec le positional embedding : "
                f"{x.shape[1]} contre {self.pos_embed.shape[1]}."
            )
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def interpolate_positional_embedding(
    pos_embed: torch.Tensor,
    target_grid_h: int,
    target_grid_w: int,
) -> torch.Tensor:
    if pos_embed.ndim != 3 or pos_embed.shape[1] != 14 * 14:
        raise RuntimeError(
            f"Positional embedding incompatible : {tuple(pos_embed.shape)}"
        )
    embedding_dim = pos_embed.shape[-1]
    pos = pos_embed.reshape(1, 14, 14, embedding_dim).permute(0, 3, 1, 2)
    pos = F.interpolate(
        pos,
        size=(target_grid_h, target_grid_w),
        mode="bicubic",
        align_corners=False,
    )
    return pos.permute(0, 2, 3, 1).reshape(
        1, target_grid_h * target_grid_w, embedding_dim
    )


class NB6ViTBackbone(nn.Module):
    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit
        self.out_channels = EMBED_DIM

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        if (height, width) != (INPUT_HEIGHT, INPUT_WIDTH):
            raise RuntimeError(
                f"Résolution reçue {width}x{height}; "
                f"résolution attendue {INPUT_WIDTH}x{INPUT_HEIGHT}."
            )
        x = self.vit.patch_embed(images)
        expected_tokens = (INPUT_HEIGHT // PATCH_SIZE) * (
            INPUT_WIDTH // PATCH_SIZE
        )
        if x.shape != (batch, expected_tokens, EMBED_DIM):
            raise RuntimeError(f"Patch embedding inattendu : {tuple(x.shape)}")
        pos = interpolate_positional_embedding(
            self.vit.pos_embed,
            INPUT_HEIGHT // PATCH_SIZE,
            INPUT_WIDTH // PATCH_SIZE,
        )
        x = x + pos.to(device=x.device, dtype=x.dtype)
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)
        return x.transpose(1, 2).reshape(
            batch,
            EMBED_DIM,
            INPUT_HEIGHT // PATCH_SIZE,
            INPUT_WIDTH // PATCH_SIZE,
        )


class NB6SimpleFeaturePyramid(nn.Module):
    def __init__(self, vit_backbone: NB6ViTBackbone, out_channels: int = 256):
        super().__init__()
        self.body = vit_backbone
        self.out_channels = out_channels
        self.proj = nn.Conv2d(EMBED_DIM, out_channels, kernel_size=1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.fpn = FeaturePyramidNetwork(
            [out_channels] * 5, out_channels=out_channels
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        base = self.body(x)
        p1 = self.proj(base)
        features = OrderedDict(
            {
                "0": self.up(p1),
                "1": p1,
                "2": self.down1(p1),
            }
        )
        features["3"] = self.down2(features["2"])
        features["4"] = self.down3(features["3"])
        return self.fpn(features)


class E2SimpleFeaturePyramid(nn.Module):
    """Pyramide P2-P6 utilisée par le checkpoint final E2/NB6.

    Les noms des attributs sont volontairement identiques à ceux du notebook
    de comparaison, car ils font partie des clés sauvegardées dans le state dict.
    """

    def __init__(self, vit: VisionTransformer, out_channels: int = 256) -> None:
        super().__init__()
        self.vit = vit
        self.out_channels = out_channels
        self.p2_up_1 = nn.ConvTranspose2d(
            EMBED_DIM, EMBED_DIM // 2, kernel_size=2, stride=2
        )
        self.p2_up_2 = nn.ConvTranspose2d(
            EMBED_DIM // 2, out_channels, kernel_size=2, stride=2
        )
        self.p2_norm = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p3 = nn.Sequential(
            nn.ConvTranspose2d(
                EMBED_DIM, out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(EMBED_DIM, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(
                EMBED_DIM, out_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p6 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        tokens = self.vit(images)
        batch = tokens.shape[0]
        base = tokens.transpose(1, 2).reshape(
            batch,
            EMBED_DIM,
            INPUT_HEIGHT // PATCH_SIZE,
            INPUT_WIDTH // PATCH_SIZE,
        )
        p2 = F.gelu(self.p2_up_1(base))
        p2 = self.p2_norm(self.p2_up_2(p2))
        p5 = self.p5(base)
        return OrderedDict(
            {
                "0": p2,
                "1": self.p3(base),
                "2": self.p4(base),
                "3": p5,
                "4": self.p6(p5),
            }
        )


class ImageNetSimpleFeaturePyramid(nn.Module):
    """Pyramide exacte du Notebook 2 utilisant un ViT fourni par timm."""

    def __init__(self, vit: nn.Module, out_channels: int = 256) -> None:
        super().__init__()
        self.vit = vit
        self.out_channels = out_channels
        self.p2_up_1 = nn.ConvTranspose2d(EMBED_DIM, EMBED_DIM // 2, 2, stride=2)
        self.p2_up_2 = nn.ConvTranspose2d(EMBED_DIM // 2, out_channels, 2, stride=2)
        self.p2_norm = nn.Sequential(nn.GroupNorm(32, out_channels), nn.GELU())
        self.p3 = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, out_channels, 2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(EMBED_DIM, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(EMBED_DIM, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )
        self.p6 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    @staticmethod
    def tokens_to_map(features: torch.Tensor, height: int, width: int) -> torch.Tensor:
        grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE
        expected = grid_h * grid_w
        if features.ndim == 4:
            if features.shape[-1] == EMBED_DIM:
                features = features.permute(0, 3, 1, 2).contiguous()
            return features
        if features.ndim != 3:
            raise RuntimeError(f"Sortie ViT ImageNet inattendue : {tuple(features.shape)}")
        if features.shape[1] == expected + 1:
            features = features[:, 1:, :]
        elif features.shape[1] != expected:
            raise RuntimeError(
                f"Tokens ImageNet incompatibles : {features.shape[1]}, attendu {expected} ou {expected + 1}."
            )
        batch, _, dimension = features.shape
        return features.transpose(1, 2).reshape(batch, dimension, grid_h, grid_w).contiguous()

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        tokens = self.vit.forward_features(images)
        base = self.tokens_to_map(tokens, images.shape[-2], images.shape[-1])
        p2 = self.p2_norm(self.p2_up_2(F.gelu(self.p2_up_1(base))))
        p5 = self.p5(base)
        return OrderedDict(
            {
                "0": p2,
                "1": self.p3(base),
                "2": self.p4(base),
                "3": p5,
                "4": self.p6(p5),
            }
        )


def build_model(
    architecture: str = "e2",
    backbone_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
) -> FasterRCNN:
    if architecture == "e2":
        vit = VisionTransformer(rectangular_pos_embed=True)
        backbone = E2SimpleFeaturePyramid(vit, out_channels=256)
    elif architecture == "nb6_legacy":
        vit = VisionTransformer(rectangular_pos_embed=False)
        backbone = NB6SimpleFeaturePyramid(NB6ViTBackbone(vit), out_channels=256)
    elif architecture == "imagenet":
        vit = timm.create_model(
            backbone_name,
            pretrained=False,
            img_size=(INPUT_HEIGHT, INPUT_WIDTH),
            num_classes=0,
            global_pool="",
        )
        backbone = ImageNetSimpleFeaturePyramid(vit, out_channels=256)
    else:
        raise ValueError(f"Architecture inconnue : {architecture}")
    anchors = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "4"],
        output_size=7,
        sampling_ratio=2,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=len(CLASS_NAMES),
        rpn_anchor_generator=anchors,
        box_roi_pool=roi_pooler,
        min_size=INPUT_HEIGHT,
        max_size=INPUT_WIDTH,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
        box_detections_per_img=300,
    )


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "detector_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Le checkpoint ne contient ni model_state_dict, ni detector_state_dict, "
        "ni state_dict compatible."
    )


def _remove_common_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_model(checkpoint_path: str | Path, device: torch.device) -> FasterRCNN:
    """Construit NB6, charge son checkpoint et place le modèle en évaluation."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}. Copiez best_detector.pt dans "
            "le dossier models/."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # Repli réservé au checkpoint historique produit par les notebooks.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = _extract_state_dict(checkpoint)
    for prefix in ("module.", "detector.", "model."):
        state_dict = _remove_common_prefix(state_dict, prefix)

    keys = tuple(state_dict.keys())
    if any(key == "backbone.vit.cls_token" for key in keys):
        architecture = "imagenet"
    elif any(key.startswith("backbone.body.vit.") for key in keys):
        architecture = "nb6_legacy"
    elif any(key.startswith("backbone.p2_up_1.") for key in keys):
        architecture = "e2"
    else:
        raise RuntimeError(
            "Architecture non reconnue dans le checkpoint. Les clés attendues "
            "pour la pyramide E2/NB6 sont absentes."
        )

    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, Mapping) else {}
    backbone_name = (
        checkpoint.get("backbone_name")
        if isinstance(checkpoint, Mapping)
        else None
    ) or metadata.get("backbone_name", "vit_small_patch16_224.augreg_in21k_ft_in1k")
    model = build_model(architecture, backbone_name=backbone_name)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Le checkpoint est incompatible avec l'architecture NB6 "
            "(ViT-S/16 + Simple Feature Pyramid + Faster R-CNN). "
            f"Détail PyTorch : {exc}"
        ) from exc

    del state_dict
    del checkpoint
    gc.collect()
    model.to(device)
    model.eval()
    return model
