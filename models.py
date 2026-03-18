from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from utils import masked_mean, masked_std


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, module: nn.Module, dropout: float, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.dropout(self.module(self.norm(x), *args, **kwargs))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout)
        self.attn_block = PreNormResidual(dim, self._attn_forward, dropout, use_rmsnorm)
        self.ff_block = PreNormResidual(dim, self.ff, dropout, use_rmsnorm)

    def _attn_forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        out, _ = self.attn(x, context, context, key_padding_mask=key_padding_mask, need_weights=False)
        return out

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn_block(x, context=context, key_padding_mask=key_padding_mask)
        x = self.ff_block(x)
        return x


class DropletFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int, fourier_features: int = 2) -> None:
        super().__init__()
        self.fourier_features = fourier_features
        base_dim = 6 + 2 * fourier_features
        self.proj = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, volume_fractions: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = 1e-8
        f = volume_fractions.clamp_min(eps)
        y = labels
        sqrt_f = torch.sqrt(f)
        log_f = torch.log(f)
        mean_f = masked_mean(f.unsqueeze(-1), mask, dim=1).squeeze(-1)
        std_f = masked_std(f.unsqueeze(-1), mask, dim=1).squeeze(-1)
        standardized = (f - mean_f.unsqueeze(1)) / std_f.unsqueeze(1).clamp_min(eps)
        feats = [f, log_f, y, f * y, sqrt_f, standardized]
        for k in range(self.fourier_features):
            freq = (2.0**k) * math.pi
            feats.append(torch.sin(freq * f))
            feats.append(torch.cos(freq * f))
        token_features = torch.stack(feats, dim=-1)
        return self.proj(token_features)


class VolumeAwareSetTransformerRegressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_latents: int = 16,
        num_heads: int = 4,
        num_self_attn_layers: int = 3,
        dropout: float = 0.1,
        fourier_features: int = 2,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = DropletFeatureEncoder(hidden_dim=hidden_dim, fourier_features=fourier_features)
        self.input_proj = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = MultiHeadAttentionBlock(latent_dim, num_heads, dropout, use_rmsnorm)
        self.latent_blocks = nn.ModuleList(
            [MultiHeadAttentionBlock(latent_dim, num_heads, dropout, use_rmsnorm) for _ in range(num_self_attn_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        volume_fractions: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = self.encoder(volume_fractions, labels, mask)
        tokens = self.input_proj(tokens)
        batch_size = tokens.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None if mask is None else ~mask
        latents = self.cross_attn(latents, context=tokens, key_padding_mask=key_padding_mask)
        for block in self.latent_blocks:
            latents = block(latents)
        pooled = latents.mean(dim=1)
        pred_log_copies = self.readout(pooled).squeeze(-1)
        pred_copies = torch.expm1(torch.clamp(pred_log_copies, min=0.0))
        return {
            "pred_log_copies": pred_log_copies,
            "pred_copies": pred_copies,
            "embedding": pooled,
        }
