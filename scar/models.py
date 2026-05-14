from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from .idm_backbone import (
    SpaceTimeIDMBackbone,
    IDMOutput,
    compute_perplexity,
    get_pos_encoding,
    patchify,
)
from .runtime import select_rgb_channels


class LatentSpaceIDM(SpaceTimeIDMBackbone):
    _LOGVAR_BIAS_INIT = -4.0

    def __init__(
        self,
        cfg,
        input_dim: tuple[int, int, int],
        la_dim: int,
        *,
        temporal_stride: int,
    ):
        super().__init__(cfg, input_dim=input_dim, la_dim=la_dim)
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        self.temporal_stride = int(temporal_stride)
        if self.cfg.net.pos_enc == "learned":
            self.spatial_pos_embed = get_pos_encoding(
                self.cfg.net.pos_enc,
                embedding_dim=self.model_dim,
                max_len=self.num_patches + 1,
            )
        self.use_reparameterized_la = bool(getattr(self.cfg, "use_reparameterized_la", False))
        self.first_la_head = nn.Linear(self.model_dim, self.la_dim)
        self.future_block_la_head = nn.Linear(
            self.model_dim,
            self.temporal_stride * self.la_dim,
        )
        if self.use_reparameterized_la:
            self.first_la_logvar_head = nn.Linear(self.model_dim, self.la_dim)
            self.future_block_la_logvar_head = nn.Linear(
                self.model_dim,
                self.temporal_stride * self.la_dim,
            )
            nn.init.constant_(self.first_la_logvar_head.bias, self._LOGVAR_BIAS_INIT)
            nn.init.constant_(self.future_block_la_logvar_head.bias, self._LOGVAR_BIAS_INIT)
        else:
            self.first_la_logvar_head = None
            self.future_block_la_logvar_head = None

    def _reshape_future_action_blocks(
        self,
        future_actions: torch.Tensor,
        *,
        batch_size: int,
        latent_steps: int,
    ) -> torch.Tensor:
        return future_actions.view(
            batch_size,
            (latent_steps - 1) * self.temporal_stride,
            self.la_dim,
        )

    def _sample_latent_actions(
        self,
        la_mean: torch.Tensor,
        la_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        la_logvar = la_logvar.clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * la_logvar)
        sampled = la_mean + torch.randn_like(std) * std
        kl = 0.5 * (la_mean.pow(2) + la_logvar.exp() - 1.0 - la_logvar)
        return sampled, la_logvar, kl.mean()

    def forward(
        self,
        observations,
        timesteps: torch.Tensor,
        states: torch.Tensor | None = None,
        *,
        target_frame_tokens: int | None = None,
        **kwargs,
    ) -> IDMOutput:
        del states, kwargs
        batch_size, latent_steps, *_ = observations.shape
        observations = observations.permute(0, 1, 3, 4, 2)
        patches = patchify(observations, self.cfg.patch_size)
        patches_embed = self.activation(self.input_embed(patches))

        if self.cfg.add_action_token:
            action_pad = self.action_in.expand(batch_size, latent_steps, 1, self.model_dim)
            patches_embed = torch.cat([action_pad, patches_embed], dim=2)

        num_tokens = patches_embed.shape[2]
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps.long()).unsqueeze(2).expand(
                batch_size,
                latent_steps,
                num_tokens,
                self.model_dim,
            )
            spatial_coord = torch.arange(num_tokens, device=patches_embed.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long()).view(
                1,
                1,
                num_tokens,
                self.model_dim,
            ).expand(batch_size, latent_steps, num_tokens, self.model_dim)
            pos_embed = spatial_pos_embed + t_pos_embed
        else:
            pos_embed = None

        z = self.encoder(
            patches_embed,
            pos_embed=pos_embed,
            causal=bool(getattr(self.cfg, "causal", False)),
        )
        z = z.view(batch_size, latent_steps, -1, self.model_dim)
        if self.cfg.add_action_token:
            la_z = z[:, :, 0]
        else:
            la_z = z.mean(dim=2)

        first_la_mean = self.first_la_head(la_z[:, :1])
        if latent_steps > 1:
            future_la_mean = self.future_block_la_head(la_z[:, 1:])
            future_la_mean = self._reshape_future_action_blocks(
                future_la_mean,
                batch_size=batch_size,
                latent_steps=latent_steps,
            )
            la_mean = torch.cat([first_la_mean, future_la_mean], dim=1)
        else:
            la_mean = first_la_mean

        la_logvar = None
        kl_loss = None
        if self.use_reparameterized_la:
            first_la_logvar = self.first_la_logvar_head(la_z[:, :1])
            if latent_steps > 1:
                future_la_logvar = self.future_block_la_logvar_head(la_z[:, 1:])
                future_la_logvar = self._reshape_future_action_blocks(
                    future_la_logvar,
                    batch_size=batch_size,
                    latent_steps=latent_steps,
                )
                la_logvar = torch.cat([first_la_logvar, future_la_logvar], dim=1)
            else:
                la_logvar = first_la_logvar
            sampled_la, la_logvar, kl_loss = self._sample_latent_actions(la_mean, la_logvar)
            la = sampled_la if self.training else la_mean
        else:
            la = la_mean

        if target_frame_tokens is not None:
            if la.shape[1] < target_frame_tokens:
                raise ValueError(
                    "LatentSpaceIDM produced fewer frame-rate actions than requested: "
                    f"have {la.shape[1]}, need {target_frame_tokens}"
                )
            la = la[:, :target_frame_tokens]
            la_mean = la_mean[:, :target_frame_tokens]
            if la_logvar is not None:
                la_logvar = la_logvar[:, :target_frame_tokens]
                kl = 0.5 * (la_mean.pow(2) + la_logvar.exp() - 1.0 - la_logvar)
                kl_loss = kl.mean()

        if self.cfg.quantize_la:
            quantized_la, indices, vq_loss = self.vq(la)
            if self.cfg.use_quantized_las:
                la = quantized_la
            vq_loss = vq_loss.mean()
            vq_outputs = {"indices": indices}
            vq_metrics = {
                "vq_loss": vq_loss.item(),
                "perplexity": compute_perplexity(
                    indices,
                    self.cfg.vq.kwargs.codebook_size,
                ),
            }
            return IDMOutput(
                la=la,
                la_mean=la_mean,
                la_logvar=la_logvar,
                kl_loss=kl_loss,
                quantized_la=quantized_la,
                vq_loss=vq_loss,
                vq_metrics=vq_metrics,
                vq_outputs=vq_outputs,
                encoder_out=patches,
            )

        return IDMOutput(
            la=la,
            la_mean=la_mean,
            la_logvar=la_logvar,
            kl_loss=kl_loss,
            encoder_out=patches,
        )


class LatentActionToGTActionLinearHead(nn.Module):
    def __init__(self, latent_action_dim: int, gt_action_dim: int):
        super().__init__()
        self.head_type = "linear"
        self.proj = nn.Linear(latent_action_dim, gt_action_dim)

    def forward(
        self,
        latent_actions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        return self.proj(latent_actions)


def _build_sinusoidal_positional_encoding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half_dim = max(1, dim // 2)
    div_term = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )
    angles = position * div_term.unsqueeze(0)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


def build_local_timesteps(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
        batch_size,
        seq_len,
    )


def prepare_latent_idm_inputs(
    idm_module: nn.Module,
    video_lat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    idm_param = next(idm_module.parameters())
    observations = rearrange(video_lat, "b c t h w -> b t c h w").to(
        device=idm_param.device,
        dtype=idm_param.dtype,
    )
    timesteps = build_local_timesteps(
        batch_size=observations.shape[0],
        seq_len=observations.shape[1],
        device=idm_param.device,
    )
    return observations, timesteps


def prepare_rgb_patch_idm_inputs(
    idm_module: nn.Module,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    idm_param = next(idm_module.parameters())
    rgb_observations = select_rgb_channels(observations)
    rgb_observations = rgb_observations.to(
        device=idm_param.device,
        dtype=idm_param.dtype,
    )
    timesteps = build_local_timesteps(
        batch_size=rgb_observations.shape[0],
        seq_len=rgb_observations.shape[1],
        device=idm_param.device,
    )
    return rgb_observations, timesteps


def resolve_gt_action_transformer_nheads(model_dim: int) -> int:
    for candidate in (4, 8, 2, 1):
        if model_dim % candidate == 0:
            return candidate
    return 1


class LatentActionToGTActionTransformerHead(nn.Module):
    def __init__(
        self,
        latent_action_dim: int,
        gt_action_dim: int,
        *,
        model_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        pos_enc: str = "learned",
        max_len: int = 512,
    ):
        super().__init__()
        self.head_type = "transformer"
        self.model_dim = int(model_dim or latent_action_dim)
        self.num_layers = int(num_layers)
        self.n_heads = resolve_gt_action_transformer_nheads(self.model_dim)
        self.dim_feedforward = self.model_dim * 4
        self.dropout = float(dropout)
        self.pos_enc = pos_enc
        self.max_len = int(max_len)

        self.input_proj = (
            nn.Identity()
            if latent_action_dim == self.model_dim
            else nn.Linear(latent_action_dim, self.model_dim)
        )
        self.input_norm = nn.LayerNorm(self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.pos_embed = get_pos_encoding(
            self.pos_enc,
            embedding_dim=self.model_dim,
            max_len=self.max_len,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, gt_action_dim)

    def forward(
        self,
        latent_actions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = latent_actions.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"GT-action transformer max_len={self.max_len} is too small for seq_len={seq_len}."
            )

        x = self.input_proj(latent_actions)
        x = self.input_norm(x)
        timesteps = build_local_timesteps(
            batch_size=batch_size,
            seq_len=seq_len,
            device=latent_actions.device,
        )
        if self.pos_enc == "learned":
            pos = self.pos_embed(timesteps.long()).to(dtype=x.dtype)
        else:
            pos = self.pos_embed.to(device=x.device, dtype=x.dtype)[timesteps]
        x = x + pos

        key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.ndim != 2:
                raise ValueError(
                    f"Expected padding_mask shape [B, T], got {tuple(padding_mask.shape)}"
                )
            key_padding_mask = ~padding_mask.to(device=x.device, dtype=torch.bool)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.output_norm(x)
        return self.output_proj(x)


class ContextGTActionToLatentActionTransformer(nn.Module):
    ARCH_SUMMARY_ADD = "summary_add"
    ARCH_LATENT_RESIDUAL_CROSS_ATTENTION = "latent_residual_cross_attention"
    ARCH_POINTWISE_MLP = "pointwise_mlp"

    def __init__(
        self,
        context_latent_dim: int,
        gt_action_dim: int,
        latent_action_dim: int,
        *,
        dim_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        architecture: str = ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
    ) -> None:
        super().__init__()
        architecture = str(architecture or self.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION)
        supported_architectures = {
            self.ARCH_SUMMARY_ADD,
            self.ARCH_LATENT_RESIDUAL_CROSS_ATTENTION,
            self.ARCH_POINTWISE_MLP,
        }
        if architecture not in supported_architectures:
            raise ValueError(
                f"Unsupported latent-action controller architecture: {architecture}. "
                f"Supported: {sorted(supported_architectures)}"
            )
        self.context_latent_dim = int(context_latent_dim)
        self.gt_action_dim = int(gt_action_dim)
        self.latent_action_dim = int(latent_action_dim)
        self.dim_model = int(dim_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.architecture = architecture

        self.context_proj = nn.Linear(context_latent_dim, dim_model)
        self.context_norm = nn.LayerNorm(dim_model)

        if self.architecture == self.ARCH_SUMMARY_ADD:
            self.action_proj = nn.Linear(gt_action_dim, dim_model)
            self.action_norm = nn.LayerNorm(dim_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_norm = nn.LayerNorm(dim_model)
            self.output_proj = nn.Linear(dim_model, latent_action_dim)
        elif self.architecture == self.ARCH_POINTWISE_MLP:
            self.pointwise_input_proj = nn.Linear(gt_action_dim, dim_model)
            self.pointwise_input_norm = nn.LayerNorm(dim_model)
            self.pointwise_mlp = nn.Sequential(
                nn.Linear(dim_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, latent_action_dim),
            )
        else:
            self.action_to_latent = nn.Linear(gt_action_dim, latent_action_dim)
            self.latent_action_proj = (
                nn.Identity()
                if latent_action_dim == dim_model
                else nn.Linear(latent_action_dim, dim_model)
            )
            self.latent_action_norm = nn.LayerNorm(dim_model)
            context_encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                context_encoder_layer,
                num_layers=max(1, n_layers),
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
            self.output_norm = nn.LayerNorm(dim_model)
            self.residual_proj = nn.Linear(dim_model, latent_action_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(
                "Expected context_tokens to have shape [B, T_ctx, C_ctx], got "
                f"{tuple(context_tokens.shape)}"
            )
        if gt_actions.ndim != 3:
            raise ValueError(
                f"Expected gt_actions to have shape [B, T, D], got {tuple(gt_actions.shape)}"
            )

        context_embed = self.context_norm(self.context_proj(context_tokens))
        if self.architecture == self.ARCH_SUMMARY_ADD:
            action_embed = self.action_norm(self.action_proj(gt_actions))
            context_summary = context_embed.mean(dim=1, keepdim=True)

            x = action_embed + context_summary
            x = x + _build_sinusoidal_positional_encoding(
                x.shape[1],
                x.shape[2],
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
            x = self.encoder(x)
            x = self.output_norm(x)
            return self.output_proj(x)
        if self.architecture == self.ARCH_POINTWISE_MLP:
            del context_embed
            x = self.pointwise_input_norm(self.pointwise_input_proj(gt_actions))
            return self.pointwise_mlp(x)

        context_embed = context_embed + _build_sinusoidal_positional_encoding(
            context_embed.shape[1],
            context_embed.shape[2],
            device=context_embed.device,
            dtype=context_embed.dtype,
        ).unsqueeze(0)
        context_memory = self.context_encoder(context_embed)

        action_latent_base = self.action_to_latent(gt_actions)
        action_tokens = self.latent_action_norm(self.latent_action_proj(action_latent_base))
        action_tokens = action_tokens + _build_sinusoidal_positional_encoding(
            action_tokens.shape[1],
            action_tokens.shape[2],
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        ).unsqueeze(0)
        decoded = self.decoder(tgt=action_tokens, memory=context_memory)
        latent_residual = self.residual_proj(self.output_norm(decoded))
        return action_latent_base + latent_residual


def build_gt_action_head(
    *,
    head_type: str,
    latent_action_dim: int,
    gt_action_dim: int,
) -> nn.Module:
    if head_type == "linear":
        return LatentActionToGTActionLinearHead(latent_action_dim, gt_action_dim)
    if head_type == "transformer":
        return LatentActionToGTActionTransformerHead(latent_action_dim, gt_action_dim)
    raise ValueError(f"Unsupported gt_action_head_type: {head_type}")


def describe_gt_action_head(action_head: nn.Module | None, unwrap_module_fn) -> str:
    action_head = unwrap_module_fn(action_head)
    if action_head is None:
        return "disabled"
    head_type = getattr(action_head, "head_type", action_head.__class__.__name__)
    if head_type == "transformer":
        return (
            f"type=transformer, model_dim={action_head.model_dim}, "
            f"layers={action_head.num_layers}, n_heads={action_head.n_heads}, "
            f"ff_dim={action_head.dim_feedforward}, dropout={action_head.dropout:.2f}, "
            f"pos_enc={action_head.pos_enc}"
        )
    return "type=linear"


def build_gt_action_padding_mask(
    mask: torch.Tensor | None,
    *,
    target_seq_len: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    seq_mask = mask[:, :target_seq_len]
    if seq_mask.ndim == 3:
        if seq_mask.dtype == torch.bool:
            seq_mask = seq_mask.any(dim=-1)
        else:
            seq_mask = seq_mask.abs().sum(dim=-1) > 0
    elif seq_mask.ndim != 2:
        raise ValueError(
            "GT-action auxiliary padding mask must have shape [B, T] or [B, T, D]; "
            f"got {tuple(seq_mask.shape)}"
        )
    return seq_mask


__all__ = [
    "LatentActionToGTActionLinearHead",
    "LatentActionToGTActionTransformerHead",
    "ContextGTActionToLatentActionTransformer",
    "LatentSpaceIDM",
    "build_gt_action_head",
    "build_gt_action_padding_mask",
    "build_local_timesteps",
    "describe_gt_action_head",
    "prepare_latent_idm_inputs",
    "prepare_rgb_patch_idm_inputs",
    "resolve_gt_action_transformer_nheads",
]
