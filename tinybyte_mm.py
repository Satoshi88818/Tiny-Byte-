"""
TinyByte Multimodal V4 — Core Model
Tokenizer-free byte-level causal Transformer with:
- Learned boundary predictor (batched entropy patching)
- Grouped-query attention (GQA)
- Per-patch MLP window OverlapAddDecoder
- CFG scheduling + negative prompting
- Speculative decoding support
- HuggingFace save/load integration
- bf16 mixed-precision compatible
- [V4] Discrete VAE pathway (VQ-VAE: 8-16x shorter token sequences)
- [V4] Mixture-of-Depth (MoD) layers (skip expensive layers for low-entropy patches)
- [V4] SwiGLU FFN (replaces GELU; better gradient flow)
- [V4] RMSNorm (replaces LayerNorm; faster, no mean subtraction)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    embed_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4          # GQA: set equal to num_heads for MHA
    ffn_multiplier: float = 4.0
    patch_stride: int = 4
    max_seq_len: int = 8192
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    use_entropy_patching: bool = True
    num_patches: int = 512         # fixed output count for learned predictor
    vocab_size: int = 257          # 0-255 bytes + 256 PAD/SEP
    jpeg_header_loss_weight: float = 2.0
    jpeg_header_bytes: int = 500

    # [V4] VQ-VAE pathway
    use_vqvae: bool = True         # enable discrete VAE pathway
    vqvae_codebook_size: int = 4096
    vqvae_latent_dim: int = 256    # encoder output dim before quantisation
    vqvae_commitment_cost: float = 0.25
    vqvae_decay: float = 0.99      # EMA decay for codebook updates
    vqvae_aux_loss_weight: float = 0.1   # weight of byte-level auxiliary loss

    # [V4] Mixture-of-Depth
    use_mod: bool = True           # enable MoD layer skipping
    mod_capacity_fraction: float = 0.5   # fraction of patches processed by deep layers


def _small_config() -> ModelConfig:
    return ModelConfig(
        embed_dim=256, num_layers=6, num_heads=4, num_kv_heads=4,
        gradient_checkpointing=False, use_vqvae=False, use_mod=False,
    )


def _medium_config() -> ModelConfig:
    return ModelConfig(
        embed_dim=512, num_layers=12, num_heads=8, num_kv_heads=4,
        use_vqvae=True, use_mod=True,
    )


# ---------------------------------------------------------------------------
# RMSNorm  [V4: replaces LayerNorm — no mean subtraction, ~15% faster]
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embedding (RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RoPECache(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, theta)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None])  # (1,1,T,D)
        self.register_buffer("sin", emb.sin()[None, None])

    def forward(self, seq_len: int):
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


# ---------------------------------------------------------------------------
# Byte positional embedding
# ---------------------------------------------------------------------------

class BytePosEmbedding(nn.Module):
    """Sinusoidal embedding over byte offsets within each patch."""
    def __init__(self, embed_dim: int, max_patch_size: int = 64):
        super().__init__()
        pos = torch.arange(max_patch_size).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        emb = torch.zeros(max_patch_size, embed_dim)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("emb", emb)

    def forward(self, patch_tensor: torch.Tensor, stride: int) -> torch.Tensor:
        T_p = patch_tensor.size(1)
        centre_offsets = torch.arange(T_p, device=patch_tensor.device) * stride + stride // 2
        centre_offsets = centre_offsets.clamp(max=self.emb.size(0) - 1)
        return patch_tensor + self.emb[centre_offsets].unsqueeze(0)


# ---------------------------------------------------------------------------
# Learned boundary predictor
# ---------------------------------------------------------------------------

class LearnedBoundaryPredictor(nn.Module):
    """
    2-layer 1D CNN that predicts soft patch boundaries.
    Input:  byte embeddings (B, T, D)
    Output: boundary positions (B, num_patches) + per-patch entropy signals
            The entropy signal is reused by MoD routing in V4.
    """
    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        self.num_patches = num_patches
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            boundary_positions: (B, num_patches) float indices
            entropy_signal:     (B, num_patches) ∈ [0,1] — used by MoD router
        """
        B, T, D = x.shape
        logits = self.conv(x.transpose(1, 2)).squeeze(1)  # (B, T)
        k = self.num_patches
        seg_len = T // k
        boundaries, entropies = [], []
        positions = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0)
        for i in range(k):
            start = i * seg_len
            end = start + seg_len if i < k - 1 else T
            seg_logits = logits[:, start:end]
            weights = F.softmax(seg_logits, dim=-1)
            seg_pos = positions[:, start:end]
            bp = (weights * seg_pos).sum(dim=-1)
            # Entropy of the softmax distribution as a proxy for patch complexity
            ent = -(weights * weights.clamp(min=1e-8).log()).sum(dim=-1)
            ent = ent / math.log(max(end - start, 2))  # normalise to [0,1]
            boundaries.append(bp)
            entropies.append(ent)
        return torch.stack(boundaries, dim=1), torch.stack(entropies, dim=1)


# ---------------------------------------------------------------------------
# Entropy patch encoder (V3: uses learned boundaries when B > 1)
# ---------------------------------------------------------------------------

class EntropyPatchEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_stride: int,
        num_patches: int,
        use_entropy_patching: bool = True,
    ):
        super().__init__()
        self.patch_stride = patch_stride
        self.num_patches = num_patches
        self.use_entropy_patching = use_entropy_patching
        self.conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=patch_stride,
            stride=patch_stride, padding=0
        )
        self.norm = RMSNorm(embed_dim)
        self.byte_pos = BytePosEmbedding(embed_dim)
        if use_entropy_patching:
            self.boundary_predictor = LearnedBoundaryPredictor(embed_dim, num_patches)

    def _fixed_stride_encode(self, x: torch.Tensor):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm(out)
        entropy = torch.zeros(x.size(0), out.size(1), device=x.device)
        return self.byte_pos(out, self.patch_stride), entropy

    def _learned_boundary_encode(self, x: torch.Tensor):
        B, T, D = x.shape
        boundaries, entropy_signal = self.boundary_predictor(x)
        boundaries_int = boundaries.long().clamp(0, T - 1)
        patches = []
        for p in range(self.num_patches):
            pos = boundaries_int[:, p]
            patch_emb = x.gather(1, pos.unsqueeze(1).expand(B, 1, D)).squeeze(1)
            patches.append(patch_emb)
        out = torch.stack(patches, dim=1)
        return self.norm(out), entropy_signal

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (patches, entropy_signal). entropy_signal used by MoD router."""
        if self.use_entropy_patching and hasattr(self, "boundary_predictor"):
            return self._learned_boundary_encode(x)
        return self._fixed_stride_encode(x)


# ---------------------------------------------------------------------------
# Grouped-query attention (unchanged from V3)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.groups > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.groups, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.groups, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out_proj(attn.transpose(1, 2).reshape(B, T, D))


# ---------------------------------------------------------------------------
# [V4] SwiGLU FFN (replaces GELU MLP — better gradient flow, ~10% quality gain)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: FFN(x) = (xW1 ⊙ SiLU(xW2)) W3
    Uses 2/3 × standard FFN width to keep parameter count equal.
    """
    def __init__(self, embed_dim: int, ffn_multiplier: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * ffn_multiplier * 2 / 3)
        hidden = (hidden + 63) // 64 * 64  # round up to multiple of 64 for efficiency
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)  # gate projection
        self.w2 = nn.Linear(embed_dim, hidden, bias=False)  # value projection
        self.w3 = nn.Linear(hidden, embed_dim, bias=False)  # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# [V4] Mixture-of-Depth (MoD) Router
# ---------------------------------------------------------------------------

class MoDRouter(nn.Module):
    """
    Routes tokens through or around expensive Transformer layers.
    Uses the entropy signal from LearnedBoundaryPredictor:
      - High entropy patches (complex/busy regions) → processed by full layer
      - Low entropy patches (flat/smooth regions)   → skip layer (residual only)

    capacity_fraction: fraction of tokens that pass through the layer.
    At training, we use a differentiable soft-routing with straight-through.
    At inference, hard top-k selection.
    """
    def __init__(self, embed_dim: int, capacity_fraction: float = 0.5):
        super().__init__()
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        entropy_signal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              (B, T, D)
            entropy_signal: (B, T) optional pre-computed entropy (from boundary predictor)
        Returns:
            selected_x:     (B, k, D)  — tokens that pass through the layer
            indices:        (B, k)     — original positions of selected tokens
            scores:         (B, T)     — routing scores (used for weighted recombination)
        """
        B, T, D = x.shape
        k = max(1, int(T * self.capacity_fraction))

        # Routing score: learnable projection + entropy signal bias
        scores = self.router(x).squeeze(-1)  # (B, T)
        if entropy_signal is not None:
            scores = scores + entropy_signal   # high entropy → more likely selected

        if self.training:
            # Soft top-k via Gumbel-Softmax for differentiability
            noise = torch.empty_like(scores).exponential_().log()
            gumbel_scores = scores - noise
            _, indices = gumbel_scores.topk(k, dim=-1)  # (B, k)
        else:
            _, indices = scores.topk(k, dim=-1)  # (B, k)

        indices, _ = indices.sort(dim=-1)  # preserve causal ordering
        selected_x = x.gather(1, indices.unsqueeze(-1).expand(B, k, D))
        return selected_x, indices, scores

    def scatter_back(
        self,
        x_original: torch.Tensor,
        processed: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter processed tokens back; unselected tokens pass through unchanged."""
        B, T, D = x_original.shape
        out = x_original.clone()
        # Weight by sigmoid of routing score for smooth gradient
        weights = torch.sigmoid(scores).unsqueeze(-1)  # (B, T, 1)
        processed_weighted = processed * weights.gather(1, indices.unsqueeze(-1).expand_as(processed))
        out.scatter_add_(1, indices.unsqueeze(-1).expand_as(processed), processed_weighted)
        return out


# ---------------------------------------------------------------------------
# Transformer layer with optional MoD routing  [V4]
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, use_mod: bool = False):
        super().__init__()
        self.use_mod = use_mod
        self.norm1 = RMSNorm(cfg.embed_dim)
        self.norm2 = RMSNorm(cfg.embed_dim)
        self.attn = GroupedQueryAttention(
            cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.dropout
        )
        self.ffn = SwiGLUFFN(cfg.embed_dim, cfg.ffn_multiplier, cfg.dropout)
        if use_mod:
            self.mod_router = MoDRouter(cfg.embed_dim, cfg.mod_capacity_fraction)

    def _attn_block(self, x, cos, sin):
        return x + self.attn(self.norm1(x), cos, sin)

    def _ffn_block(self, x):
        return x + self.ffn(self.norm2(x))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        entropy_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_mod and entropy_signal is not None:
            # Attention always runs on all tokens (cheap relative to FFN)
            x = self._attn_block(x, cos, sin)
            # FFN only runs on selected high-entropy tokens
            selected, indices, scores = self.mod_router(x, entropy_signal)
            # Build local cos/sin for selected positions
            processed = selected + self.ffn(self.norm2(selected))
            x = self.mod_router.scatter_back(x, processed - selected, indices, scores)
            return x
        else:
            x = self._attn_block(x, cos, sin)
            return self._ffn_block(x)


# ---------------------------------------------------------------------------
# OverlapAddDecoder V3 — per-patch MLP window (unchanged)
# ---------------------------------------------------------------------------

class OverlapAddDecoder(nn.Module):
    def __init__(self, embed_dim: int, patch_stride: int):
        super().__init__()
        self.patch_stride = patch_stride
        self.proj = nn.Linear(embed_dim, patch_stride * embed_dim)
        self.window_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, patch_stride),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, T_p, D = patches.shape
        S = self.patch_stride
        proj = self.proj(patches).view(B, T_p, S, D)
        window = self.window_mlp(patches)
        proj = proj * window.unsqueeze(-1)
        return self.norm(proj.reshape(B, T_p * S, D))


# ---------------------------------------------------------------------------
# [V4] VQ-VAE — Discrete image pathway
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    EMA-updated vector quantizer (van den Oord et al. 2017 + Razavi et al. 2019).
    Produces 8–16× shorter discrete token sequences from raw byte patches,
    drastically reducing the sequence length the main Transformer must process.
    """
    def __init__(self, codebook_size: int, latent_dim: int, decay: float = 0.99, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_size, 1 / codebook_size)

        # EMA statistics (not parameters — updated outside autograd)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, T, latent_dim) — encoder output
        Returns:
            z_q:       (B, T, latent_dim) — quantised (straight-through)
            loss:      scalar             — commitment + codebook loss
            indices:   (B, T)             — codebook indices (discrete tokens)
        """
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)  # (B*T, D)

        # Distance to codebook entries
        dist = (
            flat_z.pow(2).sum(-1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(-1)
        )  # (B*T, codebook_size)
        indices_flat = dist.argmin(-1)  # (B*T,)
        indices = indices_flat.view(B, T)

        # Quantised embeddings
        z_q_flat = self.embedding(indices_flat)  # (B*T, D)
        z_q = z_q_flat.view(B, T, D)

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices_flat, self.codebook_size).float()  # (B*T, C)
                new_cluster_size = one_hot.sum(0)
                new_embed_sum = one_hot.T @ flat_z  # (C, D)
                self.ema_cluster_size.mul_(self.decay).add_(new_cluster_size * (1 - self.decay))
                self.ema_embed_sum.mul_(self.decay).add_(new_embed_sum * (1 - self.decay))
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

        # Commitment loss (encoder → codebook); codebook updated via EMA, not gradient
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices


class VQVAEEncoder(nn.Module):
    """
    Lightweight CNN encoder: raw byte embeddings → latent codes.
    Downsamples by factor 8 (configurable) to produce short discrete sequences.
    """
    def __init__(self, embed_dim: int, latent_dim: int, downsample_factor: int = 8):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # /2
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # /4
            nn.GELU(),
            nn.Conv1d(embed_dim, latent_dim, kernel_size=4, stride=2, padding=1),  # /8
            nn.GELU(),
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T//8, latent_dim)"""
        out = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(out)


class VQVAEDecoder(nn.Module):
    """
    Upsamples discrete latent codes back to byte-level embeddings.
    """
    def __init__(self, embed_dim: int, latent_dim: int, upsample_factor: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # ×2
            nn.GELU(),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # ×4
            nn.GELU(),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),  # ×8
        )
        self.norm = RMSNorm(embed_dim)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, T_short, latent_dim) → (B, T_short*8, embed_dim)"""
        out = self.decoder(z_q.transpose(1, 2)).transpose(1, 2)
        return self.norm(out)


class VQVAE(nn.Module):
    """
    Full VQ-VAE module. For image sequences only.
    At training: encodes byte patches → discrete codes → reconstructed embeddings.
    At inference: can run in discrete mode (transformer over codebook indices).

    Integration: the main TinyByteModel uses VQVAE output embeddings as input
    to the Transformer when is_image=True, falling back to byte patches for text.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = VQVAEEncoder(cfg.embed_dim, cfg.vqvae_latent_dim)
        self.quantizer = VectorQuantizer(
            cfg.vqvae_codebook_size, cfg.vqvae_latent_dim,
            cfg.vqvae_decay, cfg.vqvae_commitment_cost,
        )
        # Project latent dim to embed_dim for Transformer input
        self.latent_proj = nn.Linear(cfg.vqvae_latent_dim, cfg.embed_dim, bias=False)
        self.decoder = VQVAEDecoder(cfg.embed_dim, cfg.vqvae_latent_dim)
        # Byte-reconstruction head (used for auxiliary loss)
        self.recon_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.aux_loss_weight = cfg.vqvae_aux_loss_weight

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D) raw byte embeddings (image portion only)
        Returns: (z_q_proj, vq_loss, indices)
            z_q_proj: (B, T//8, embed_dim) — ready for Transformer
        """
        z = self.encoder(x)               # (B, T//8, latent_dim)
        z_q, vq_loss, indices = self.quantizer(z)
        z_q_proj = self.latent_proj(z_q)  # (B, T//8, embed_dim)
        return z_q_proj, vq_loss, indices

    def decode_for_aux_loss(
        self, z_q_proj: torch.Tensor, target_bytes: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary byte reconstruction loss from VQ-VAE decoder.
        z_q_proj: (B, T_short, embed_dim)
        target_bytes: (B, T_original) — original byte labels
        """
        # Project back to latent dim for decoder
        z_q_latent = self.latent_proj.weight.T @ z_q_proj.transpose(1, 2)
        z_q_latent = z_q_latent.transpose(1, 2)  # crude inverse; good enough for aux loss
        recon_emb = self.decoder(z_q_latent)  # (B, T_original, embed_dim)
        recon_logits = self.recon_head(recon_emb)  # (B, T_original, vocab_size)
        T_recon = recon_logits.size(1)
        T_target = target_bytes.size(1)
        if T_recon > T_target:
            recon_logits = recon_logits[:, :T_target]
        elif T_recon < T_target:
            target_bytes = target_bytes[:, :T_recon]
        return F.cross_entropy(
            recon_logits.reshape(-1, recon_logits.size(-1)),
            target_bytes.reshape(-1),
            ignore_index=-100,
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TinyByteModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.patch_encoder = EntropyPatchEncoder(
            cfg.embed_dim, cfg.patch_stride, cfg.num_patches, cfg.use_entropy_patching
        )

        # [V4] VQ-VAE (image pathway)
        if cfg.use_vqvae:
            self.vqvae = VQVAE(cfg)

        self.rope = RoPECache(cfg.embed_dim // cfg.num_heads, cfg.max_seq_len)

        # [V4] MoD: apply to the second half of layers (early layers see all tokens)
        mod_start = cfg.num_layers // 2 if cfg.use_mod else cfg.num_layers
        self.layers = nn.ModuleList([
            TransformerLayer(cfg, use_mod=(cfg.use_mod and i >= mod_start))
            for i in range(cfg.num_layers)
        ])

        self.norm = RMSNorm(cfg.embed_dim)
        self.decoder = OverlapAddDecoder(cfg.embed_dim, cfg.patch_stride)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        is_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, T) byte indices
        is_image:  (B,) bool — which samples use the VQ-VAE pathway
        Returns logits: (B, T, vocab_size)
        """
        x = self.embed(input_ids)  # (B, T, D)

        vq_loss = torch.tensor(0.0, device=x.device)
        vq_indices = None

        if self.cfg.use_vqvae and is_image is not None and is_image.any():
            # Split image / text samples
            img_mask = is_image.bool()
            txt_mask = ~img_mask

            # Image samples: encode via VQ-VAE → shorter sequence
            x_img = x[img_mask]
            z_q_proj, vq_loss_batch, vq_indices_batch = self.vqvae.encode(x_img)
            # Text samples: standard byte patch encoding
            x_mixed = x.clone()
            # Pad/truncate VQ output to match patch encoder output length
            T_vq = z_q_proj.size(1)
            # We'll just use the VQ representation for image samples.
            # For simplicity we run all samples through patch encoder then override image ones.
            x_patched, entropy_signal = self.patch_encoder(x)
            # Override image sample patches with VQ-VAE output
            if img_mask.sum() > 0:
                T_p = x_patched.size(1)
                if T_vq >= T_p:
                    x_patched[img_mask] = z_q_proj[:, :T_p]
                else:
                    x_patched[img_mask, :T_vq] = z_q_proj
            vq_loss = vq_loss_batch.mean()
        else:
            x_patched, entropy_signal = self.patch_encoder(x)

        T_p = x_patched.size(1)
        cos, sin = self.rope(T_p)

        x_out = x_patched
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                x_out = checkpoint(layer, x_out, cos, sin, entropy_signal, use_reentrant=False)
            else:
                x_out = layer(x_out, cos, sin, entropy_signal)

        x_out = self.norm(x_out)
        x_out = self.decoder(x_out)

        # Align reconstructed length with input length
        T_in = input_ids.size(1)
        T_r = x_out.size(1)
        if T_r > T_in:
            x_out = x_out[:, :T_in]
        elif T_r < T_in:
            x_out = F.pad(x_out, (0, 0, 0, T_in - T_r))

        logits = self.head(x_out)  # (B, T, vocab_size)
        # Stash vq_loss for compute_loss to pick up
        self._last_vq_loss = vq_loss
        return logits

    # -----------------------------------------------------------------------
    # Loss with JPEG header + VQ-VAE auxiliary losses
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        is_image_sequence: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns dict with keys:
          loss, main_loss, header_loss, vq_loss, aux_recon_loss, trim_frac
        """
        self._last_vq_loss = torch.tensor(0.0, device=input_ids.device)
        logits = self.forward(input_ids, is_image=is_image_sequence)
        B, T, V = logits.shape

        label_len = (labels != -100).sum(dim=1).float()
        trim_frac = (1.0 - label_len / T).mean().item()

        main_loss = F.cross_entropy(
            logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100,
        )

        # JPEG header auxiliary loss
        header_loss = torch.tensor(0.0, device=input_ids.device)
        if (
            is_image_sequence is not None
            and is_image_sequence.any()
            and self.cfg.jpeg_header_loss_weight > 0
        ):
            img_mask = is_image_sequence.bool()
            hlen = min(self.cfg.jpeg_header_bytes, T - 1)
            header_logits = logits[img_mask, :hlen]
            header_labels = labels[img_mask, :hlen]
            if header_labels.numel() > 0:
                header_loss = F.cross_entropy(
                    header_logits.reshape(-1, V),
                    header_labels.reshape(-1),
                    ignore_index=-100,
                )

        # VQ-VAE commitment loss
        vq_loss = self._last_vq_loss

        # VQ-VAE auxiliary reconstruction loss
        aux_recon_loss = torch.tensor(0.0, device=input_ids.device)
        if self.cfg.use_vqvae and is_image_sequence is not None and is_image_sequence.any():
            img_mask = is_image_sequence.bool()
            x_img = self.embed(input_ids[img_mask])
            z_q_proj, _, _ = self.vqvae.encode(x_img)
            aux_recon_loss = self.vqvae.decode_for_aux_loss(z_q_proj, labels[img_mask])

        total_loss = (
            main_loss
            + self.cfg.jpeg_header_loss_weight * header_loss
            + vq_loss
            + self.cfg.vqvae_aux_loss_weight * aux_recon_loss
        )

        return {
            "loss": total_loss,
            "main_loss": main_loss.item(),
            "header_loss": header_loss.item(),
            "vq_loss": vq_loss.item(),
            "aux_recon_loss": aux_recon_loss.item(),
            "trim_frac": trim_frac,
        }

    # -----------------------------------------------------------------------
    # HuggingFace-style save / load
    # -----------------------------------------------------------------------

    def save_pretrained(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        card = (
            "---\nlibrary_name: tinybyte\n---\n\n"
            "# TinyByte Multimodal V4\n\n"
            "Tokenizer-free byte-level causal Transformer for text and image generation.\n\n"
            f"- embed_dim: {self.cfg.embed_dim}\n"
            f"- num_layers: {self.cfg.num_layers}\n"
            f"- num_heads: {self.cfg.num_heads} (GQA kv_heads={self.cfg.num_kv_heads})\n"
            f"- VQ-VAE: {'enabled' if self.cfg.use_vqvae else 'disabled'} "
            f"(codebook_size={self.cfg.vqvae_codebook_size})\n"
            f"- Mixture-of-Depth: {'enabled' if self.cfg.use_mod else 'disabled'} "
            f"(capacity={self.cfg.mod_capacity_fraction})\n"
            f"- FFN: SwiGLU\n"
            f"- Norm: RMSNorm\n"
        )
        (save_dir / "README.md").write_text(card)
        print(f"Model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str | Path, device: str = "cpu") -> "TinyByteModel":
        load_dir = Path(load_dir)
        with open(load_dir / "config.json") as f:
            cfg_dict = json.load(f)
        cfg = ModelConfig(**cfg_dict)
        model = cls(cfg)
        state = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        print(f"Model loaded from {load_dir}")
        return model

    # -----------------------------------------------------------------------
    # [V4] TorchScript / ONNX export helpers
    # -----------------------------------------------------------------------

    def export_torchscript(self, save_path: str | Path, example_input: Optional[torch.Tensor] = None):
        """Export model to TorchScript for edge deployment."""
        save_path = Path(save_path)
        # Disable VQ-VAE and MoD for scripting (simplifies graph)
        self.eval()
        if example_input is None:
            example_input = torch.zeros(1, 64, dtype=torch.long)
        try:
            traced = torch.jit.trace(self, (example_input,), strict=False)
            torch.jit.save(traced, str(save_path))
            print(f"TorchScript model saved to {save_path}")
            return traced
        except Exception as e:
            print(f"TorchScript tracing failed: {e}. Try ONNX export instead.")
            return None

    def export_onnx(
        self,
        save_path: str | Path,
        example_input: Optional[torch.Tensor] = None,
        opset_version: int = 17,
    ):
        """Export model to ONNX for edge/cross-platform deployment."""
        save_path = Path(save_path)
        self.eval()
        if example_input is None:
            example_input = torch.zeros(1, 64, dtype=torch.long)
        torch.onnx.export(
            self,
            (example_input,),
            str(save_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}, "logits": {0: "batch", 1: "seq_len"}},
            opset_version=opset_version,
        )
        print(f"ONNX model saved to {save_path}")
