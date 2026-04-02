"""
TinyByte V4 — Speculative Decoding Draft Model with Co-Distillation

V4 changes from V3:
- Co-distillation training: DraftModel trained jointly with main model
  via knowledge distillation (KL divergence from main model logits).
  This dramatically improves acceptance rate vs independently trained draft.
- DraftModel now uses same SwiGLU FFN and RMSNorm as main model for
  consistent distribution alignment.
- DraftModelTrainer handles the co-distillation loss scheduling.
- Acceptance rate logging broken out per sequence position for diagnostics.

Usage:
    # Co-distillation training (call from main train.py loop):
    draft_trainer = DraftModelCoDistiller(main_model, draft_model)
    draft_loss = draft_trainer.distill_step(input_ids, labels)

    # Inference (unchanged API from V3):
    output = speculative_decode(main, draft, prompt_ids, max_new_bytes=1024)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinybyte_mm import ModelConfig, TinyByteModel

DRAFT_CONFIG = ModelConfig(
    embed_dim=256,
    num_layers=6,
    num_heads=4,
    num_kv_heads=4,
    gradient_checkpointing=False,
    use_entropy_patching=False,
    use_vqvae=False,    # draft stays byte-level for speed
    use_mod=False,      # draft processes all tokens
)


class DraftModel(TinyByteModel):
    """
    Lightweight 6L/256D draft model for speculative decoding.
    V4: Uses same architectural primitives as main model (SwiGLU, RMSNorm)
    for better distribution alignment during co-distillation.
    """
    def __init__(self):
        super().__init__(DRAFT_CONFIG)

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "DraftModel":
        import json
        from pathlib import Path
        load_dir = Path(load_dir)
        model = cls()
        state = torch.load(load_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        return model

    def save_pretrained(self, save_dir: str):
        from pathlib import Path
        import json
        from dataclasses import asdict
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        print(f"DraftModel saved to {save_dir}")


# ---------------------------------------------------------------------------
# [V4] Co-distillation trainer
# ---------------------------------------------------------------------------

class DraftModelCoDistiller:
    """
    Trains the draft model jointly with the main model via knowledge distillation.

    Loss = alpha * CE(draft_logits, hard_labels)
           + (1-alpha) * KL(draft_probs || main_probs.detach())

    The KL term pulls the draft distribution toward the main model's output
    distribution, dramatically improving speculative decoding acceptance rate
    without requiring the main model to expose its weights to gradient flow.

    Usage (inside main training loop):
        distiller = DraftModelCoDistiller(main_model, draft_model)
        draft_opt = torch.optim.AdamW(draft_model.parameters(), lr=1e-3)

        for batch in dataloader:
            # Normal main model training step ...
            main_opt.step()

            # Co-distillation step (no gradient into main model):
            draft_loss = distiller.distill_step(
                input_ids, labels, draft_opt, temperature=2.0
            )
    """

    def __init__(
        self,
        main_model: TinyByteModel,
        draft_model: DraftModel,
        alpha: float = 0.3,          # weight of hard label CE loss
        distill_temperature: float = 2.0,
        update_every: int = 1,       # distill every N main-model steps
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.alpha = alpha
        self.temperature = distill_temperature
        self.update_every = update_every
        self._step = 0

    @torch.no_grad()
    def _get_main_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run main model in inference mode to get soft targets."""
        self.main_model.eval()
        logits = self.main_model(input_ids)
        self.main_model.train()
        return logits

    def distill_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler=None,
    ) -> float:
        """
        Performs one co-distillation update step.
        Returns scalar loss value.
        """
        self._step += 1
        if self._step % self.update_every != 0:
            return 0.0

        self.draft_model.train()
        optimizer.zero_grad()

        B, T = input_ids.shape

        # Get soft targets from main model (no grad)
        main_logits = self._get_main_logits(input_ids)  # (B, T, V)
        main_probs = F.softmax(main_logits / self.temperature, dim=-1).detach()

        # Draft model forward
        draft_logits = self.draft_model(input_ids)  # (B, T, V)
        V = draft_logits.size(-1)

        # Hard label loss (standard CE)
        ce_loss = F.cross_entropy(
            draft_logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Soft KL divergence loss
        draft_log_probs = F.log_softmax(draft_logits / self.temperature, dim=-1)
        # Only compute KL on non-masked positions
        valid_mask = (labels != -100).reshape(-1)
        kl_loss = F.kl_div(
            draft_log_probs.reshape(-1, V)[valid_mask],
            main_probs.reshape(-1, V)[valid_mask],
            reduction="batchmean",
        ) * (self.temperature ** 2)  # scale back per Hinton et al.

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), 1.0)
            optimizer.step()

        return total_loss.item()


# ---------------------------------------------------------------------------
# Speculative decoding (V4: per-position acceptance rate logging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_decode(
    main_model: TinyByteModel,
    draft_model: DraftModel,
    prompt_ids: torch.Tensor,
    max_new_bytes: int = 512,
    draft_steps: int = 6,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Speculative decoding with per-position acceptance diagnostics.
    Returns: generated token ids (1, max_new_bytes)
    """
    main_model.eval()
    draft_model.eval()
    main_model.to(device)
    draft_model.to(device)

    generated = prompt_ids.clone().to(device)
    n_accepted_total = 0
    n_draft_total = 0
    # [V4] Per-position acceptance tracking
    position_accepts = [0] * draft_steps
    position_totals = [0] * draft_steps

    while generated.size(1) - prompt_ids.size(1) < max_new_bytes:
        # --- Draft phase ---
        draft_ids = generated.clone()
        draft_tokens, draft_probs = [], []
        for _ in range(draft_steps):
            logits = draft_model(draft_ids)[:, -1, :]
            logits = logits / temperature
            probs = _top_p_filter(F.softmax(logits, dim=-1), top_p)
            tok = torch.multinomial(probs, 1)
            draft_tokens.append(tok)
            draft_probs.append(probs[0, tok[0, 0]])
            draft_ids = torch.cat([draft_ids, tok], dim=1)

        # --- Verify phase ---
        verify_input = torch.cat([generated] + draft_tokens, dim=1)
        main_logits = main_model(verify_input)

        n_accepted = 0
        for i, (draft_tok, draft_p) in enumerate(zip(draft_tokens, draft_probs)):
            pos = generated.size(1) + i - 1
            main_logit = main_logits[:, pos, :]
            main_p_dist = _top_p_filter(F.softmax(main_logit / temperature, dim=-1), top_p)
            main_p = main_p_dist[0, draft_tok[0, 0]]
            accept_prob = torch.clamp(main_p / (draft_p + 1e-8), max=1.0)
            position_totals[i] += 1
            if torch.rand(1).item() < accept_prob.item():
                generated = torch.cat([generated, draft_tok], dim=1)
                n_accepted += 1
                position_accepts[i] += 1
            else:
                corrected = F.relu(main_p_dist - draft_p)
                corrected = corrected / corrected.sum().clamp(min=1e-8)
                tok = torch.multinomial(corrected, 1)
                generated = torch.cat([generated, tok], dim=1)
                break

        n_accepted_total += n_accepted
        n_draft_total += draft_steps

        if n_accepted == draft_steps:
            bonus_logit = main_logits[:, generated.size(1) - 1, :]
            bonus_probs = _top_p_filter(F.softmax(bonus_logit / temperature, dim=-1), top_p)
            bonus_tok = torch.multinomial(bonus_probs, 1)
            generated = torch.cat([generated, bonus_tok], dim=1)

    acceptance_rate = n_accepted_total / max(n_draft_total, 1)

    if verbose:
        print(f"Speculative decoding — overall acceptance rate: {acceptance_rate:.1%}")
        print("Per-position acceptance rates:")
        for i in range(draft_steps):
            if position_totals[i] > 0:
                pos_rate = position_accepts[i] / position_totals[i]
                print(f"  Position {i+1}: {pos_rate:.1%} ({position_accepts[i]}/{position_totals[i]})")
    else:
        print(f"Speculative decoding acceptance rate: {acceptance_rate:.1%}")

    return generated[:, prompt_ids.size(1):]


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out
