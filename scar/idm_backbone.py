from __future__ import annotations

"""IDM backbone and transformer utilities.

This module vendors the minimal legacy model pieces that the cycle pipeline needs
so scar no longer imports model code from the old package.
"""

from dataclasses import dataclass
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .idm_vq import NSVQ, SimpleNSVQ


@dataclass
class IDMOutput:
    la: torch.Tensor
    la_mean: torch.Tensor | None = None
    la_logvar: torch.Tensor | None = None
    kl_loss: torch.Tensor | None = None
    quantized_la: torch.Tensor | None = None
    vq_loss: torch.Tensor | None = None
    vq_metrics: dict | None = None
    vq_outputs: dict | None = None
    encoder_out: torch.Tensor | None = None
    state_seq: torch.Tensor | None = None
    fdm_features: torch.Tensor | None = None
    fdm_beta: torch.Tensor | None = None


def compute_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    indices_count = torch.bincount(indices.reshape(-1), minlength=codebook_size)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(indices_count)
    avg_probs = indices_count.float() / indices_count.sum()
    return (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()


def get_activation_fn(activation: str) -> Callable:
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu/glu, not {activation}.')


def patchify(videos: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w, _ = videos.shape
    padding_height = -h % size
    padding_width = -w % size
    padded = F.pad(videos, (0, 0, 0, padding_width, 0, padding_height, 0, 0, 0, 0))
    return einops.rearrange(
        padded,
        'b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)',
        hp=size,
        wp=size,
    )


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> torch.Tensor:
    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def get_pos_encoding(pos_enc_type: str, embedding_dim: int, max_len: int):
    if pos_enc_type == 'sine':
        return create_sinusoidal_pos_embedding(num_positions=max_len, dimension=embedding_dim)
    if pos_enc_type == 'learned':
        return nn.Embedding(max_len, embedding_dim)
    raise ValueError(f'Unsupported positional encoding: {pos_enc_type}')


def get_vq_cls(cls_name: str):
    if cls_name in {'simple_nsvq', 'vqNSVQ'}:
        return {'simple_nsvq': SimpleNSVQ, 'vqNSVQ': NSVQ}[cls_name]

    try:
        from vector_quantize_pytorch import FSQ, ResidualFSQ, ResidualVQ, VectorQuantize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'vector_quantize_pytorch is required for quantized latent-action backbones.'
        ) from exc

    mapping = {
        'residual_fsq': ResidualFSQ,
        'fsq': FSQ,
        'residual': ResidualVQ,
        'ema': VectorQuantize,
    }
    if cls_name not in mapping:
        raise ValueError(f'vq_cls: {cls_name} not supported')
    return mapping[cls_name]


class STBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)
        self.dropout4 = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim_model)
        self.activation = get_activation_fn(cfg.feedforward_activation)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None,
        causal: bool,
        cond: torch.Tensor | None = None,
        cond_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_tokens, _ = x.size()
        x = x.flatten(start_dim=0, end_dim=1)
        pos_embed_flat = pos_embed.flatten(start_dim=0, end_dim=1) if pos_embed is not None else None

        skip = x
        if self.cfg.pre_norm:
            x = self.norm(x)
        q = k = x if pos_embed_flat is None else x + pos_embed_flat
        x, _ = self.spatial_attn(q, k, value=x)
        x = skip + self.dropout1(x)
        if self.cfg.pre_norm:
            skip = x
            x = self.norm(x)
        else:
            x = self.norm(x)
            skip = x

        x = einops.rearrange(x, '(b t) n d -> b t n d', b=batch_size, t=seq_len)
        x = x.transpose(1, 2)
        x = x.flatten(start_dim=0, end_dim=1)
        pos_embed_flat = pos_embed.transpose(1, 2) if pos_embed is not None else None
        pos_embed_flat = pos_embed_flat.flatten(start_dim=0, end_dim=1) if pos_embed_flat is not None else None

        skip = x
        if self.cfg.pre_norm:
            x = self.norm(x)
        q = k = x if pos_embed_flat is None else x + pos_embed_flat
        if causal:
            steps = x.size(1)
            causal_mask = torch.triu(torch.ones(steps, steps, device=x.device), diagonal=1).bool()
        else:
            causal_mask = None
        x, _ = self.temporal_attn(q, k, value=x, attn_mask=causal_mask, is_causal=causal)
        x = skip + self.dropout2(x)
        if self.cfg.pre_norm:
            skip = x
            x = self.norm(x)
        else:
            x = self.norm(x)
            skip = x

        if cond is not None:
            if cond_pos_embed is None:
                raise ValueError('cond_pos_embed must be provided when cond is not None')
            cond = einops.repeat(cond, 'b t 1 d -> (b n) t d', n=num_tokens)
            skip = x
            if self.cfg.pre_norm:
                x = self.norm(x)
            q = x if pos_embed_flat is None else x + pos_embed_flat
            cond_pos_embed = einops.rearrange(cond_pos_embed, 'b t n d -> (b n) t d')
            k = cond + cond_pos_embed
            v = cond
            x, _ = self.cross_attn(q, k, value=v, attn_mask=causal_mask, is_causal=causal)
            x = skip + self.dropout3(x)
            if self.cfg.pre_norm:
                skip = x
                x = self.norm(x)
            else:
                x = self.norm(x)
                skip = x

        x = einops.rearrange(x, '(b n) t d -> b t n d', b=batch_size, t=seq_len)
        skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout4(x)
        if not self.cfg.pre_norm:
            x = self.norm(x)
        return x


class STTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(cfg=cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model) if cfg.pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        causal: bool = False,
        cond: torch.Tensor | None = None,
        cond_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                pos_embed=pos_embed,
                causal=causal,
                cond=cond,
                cond_pos_embed=cond_pos_embed,
            )
        return self.norm(x)


class SpaceTimeIDMBackbone(nn.Module):
    def __init__(self, cfg, input_dim: tuple[int, int, int], la_dim: int):
        super().__init__()
        self.name = 'SpaceTimeIDMBackbone'
        self.cfg = cfg
        self.input_dim = input_dim
        self.la_dim = la_dim
        channels, height, width = input_dim
        self.patch_token_dim = channels * self.cfg.patch_size ** 2
        self.model_dim = self.cfg.net.dim_model
        if height % self.cfg.patch_size != 0 or width % self.cfg.patch_size != 0:
            raise ValueError(
                f'Expected latent H,W divisible by patch_size={self.cfg.patch_size}, got {(height, width)}'
            )
        self.num_patches = (height // self.cfg.patch_size) * (width // self.cfg.patch_size)

        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

        self.input_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.encoder = STTransformer(cfg=self.cfg.net)
        self.activation = nn.LeakyReLU(0.2)
        self.action_in = nn.Parameter(torch.randn(1, 1, 1, self.model_dim))
        self.spatial_pos_embed = get_pos_encoding(self.cfg.net.pos_enc, self.model_dim, 200)
        self.temporal_pos_embed = get_pos_encoding(self.cfg.net.pos_enc, self.model_dim, 200)
        self.la_head = nn.Linear(self.model_dim, self.la_dim)

        self.vq = None
        if getattr(self.cfg, 'quantize_la', False):
            vq_cls = get_vq_cls(self.cfg.vq.name)
            self.cfg.vq.kwargs.dim = self.cfg.la_dim
            self.vq = vq_cls(**self.cfg.vq.kwargs)
