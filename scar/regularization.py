"""
Regularization modules for latent action disentanglement.

Currently implements:
- Gradient Reversal Layer (GRL) + Embodiment Classifier
  Encourages embodiment-agnostic latent actions via domain adversarial training.

Future:
- Add more regularization methods here for ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .metrics import MetricBundle, build_metric_bundle


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (self-contained, no external dependency)
# ---------------------------------------------------------------------------

class _GradientReversalFn(Function):
    """Reverses gradient during backprop, identity during forward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scaling factor ``alpha``."""
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    return _GradientReversalFn.apply(x, alpha)


# ---------------------------------------------------------------------------
# Embodiment Classifier (domain discriminator)
# ---------------------------------------------------------------------------

class EmbodimentClassifier(nn.Module):
    """
    MLP classifier on top of (gradient-reversed) latent actions to predict
    which embodiment produced them.

    Architecture:
        latent_actions [N, D]
            → GRL(alpha)
            → Linear → ReLU → Linear → logits [N, num_embodiments]

    The GRL ensures that minimising the classification loss w.r.t. the
    classifier weights improves discrimination, while the reversed gradient
    to the encoder (IDM) *removes* embodiment-specific information from the
    latent actions.

    Parameters
    ----------
    input_dim : int
        Latent action dimension (``la_dim``).
    num_embodiments : int
        Number of distinct embodiment classes.
    hidden_dim : int
        Hidden layer width.  Default 256.
    grl_alpha : float
        Gradient reversal scaling factor.  Can be ramped up during training.
    """

    def __init__(
        self,
        input_dim: int,
        num_embodiments: int,
        hidden_dim: int = 256,
        grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.num_embodiments = num_embodiments
        self.grl_alpha = grl_alpha
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_embodiments),
        )

    def forward(
        self,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_actions : [N, D]
            Flattened latent actions from IDM.

        Returns
        -------
        logits : [N, num_embodiments]
        """
        # Gradient reversal
        latent_actions = grad_reverse(latent_actions, self.grl_alpha)
        return self.classifier(latent_actions)

    def set_grl_alpha(self, alpha: float):
        """Update GRL scaling (e.g. ramp up during training)."""
        self.grl_alpha = alpha


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_embodiment_adversarial_loss(
    classifier: EmbodimentClassifier,
    latent_actions: torch.Tensor,
    embodiment_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, MetricBundle]:

    # print("mask:", mask.shape if mask is not None else None)
    """
    Compute cross-entropy loss for embodiment classification with GRL.

    Parameters
    ----------
    classifier : EmbodimentClassifier
    latent_actions : [B, T, D]
        Latent actions (with grad, from online IDM). Each action token,
        including the first action token, is classified independently.
    embodiment_ids : [B]
        Integer class labels (e.g. 0=franka, 1=aloha).
    mask : [B, T] or None

    Returns
    -------
    loss : scalar tensor
    metrics : dict
    """
    if latent_actions.ndim != 3:
        raise ValueError(
            f"Expected latent_actions to have shape [B, T, D], got {tuple(latent_actions.shape)}"
        )

    batch_size, seq_len, latent_dim = latent_actions.shape
    token_labels = embodiment_ids.long().view(batch_size, 1).expand(batch_size, seq_len)

    if mask is not None:
        if mask.shape[:2] != latent_actions.shape[:2]:
            raise ValueError(
                "Expected mask to have shape [B, T] aligned with latent_actions, "
                f"got {tuple(mask.shape)} for latent_actions {tuple(latent_actions.shape)}"
            )
        valid_mask = mask.to(device=latent_actions.device, dtype=torch.bool)
    else:
        valid_mask = torch.ones(
            (batch_size, seq_len),
            device=latent_actions.device,
            dtype=torch.bool,
        )

    flat_actions = latent_actions.reshape(batch_size * seq_len, latent_dim)
    flat_labels = token_labels.reshape(batch_size * seq_len)
    flat_valid_mask = valid_mask.reshape(batch_size * seq_len)

    if not bool(flat_valid_mask.any()):
        zero = latent_actions.new_zeros(())
        metrics = build_metric_bundle(
            "grl",
            grl_loss=0.0,
            grl_accuracy=0.0,
        )
        return zero, metrics

    valid_actions = flat_actions[flat_valid_mask]
    valid_labels = flat_labels[flat_valid_mask]

    logits = classifier(valid_actions)  # [N_valid, num_classes]
    loss = F.cross_entropy(logits, valid_labels)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == valid_labels).float().mean()

    metrics = build_metric_bundle(
        "grl",
        grl_loss=float(loss.detach().cpu()),
        grl_accuracy=float(accuracy.cpu()),
    )
    return loss, metrics


# ---------------------------------------------------------------------------
# GRL alpha scheduling (optional ramp-up)
# ---------------------------------------------------------------------------

def compute_grl_alpha(
    step: int,
    warmup_steps: int = 200,
    max_alpha: float = 1.0,
) -> float:
    """Linear ramp-up of GRL alpha from 0 to max_alpha over warmup_steps."""
    if step <= 0:
        return 0.0
    if warmup_steps <= 0:
        return max_alpha
    return min(step / warmup_steps, 1.0) * max_alpha


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "grad_reverse",
    "EmbodimentClassifier",
    "compute_embodiment_adversarial_loss",
    "compute_grl_alpha",
]
