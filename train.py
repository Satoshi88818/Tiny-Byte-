"""
TinyByte V4 Trainer

V4 additions over V3:
- Co-distillation of DraftModel alongside main model training
- VQ-VAE loss terms tracked separately in WandB
- MoD capacity annealing (starts at 1.0, ramps to target over warmup)
- Improved FSDP checkpoint saving (handles VQ-VAE parameters correctly)
- CC12M / LAION-COCO scale notes in comments
- --draft-codistill flag enables joint draft training
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataset import DataConfig, MultimodalByteDataset, WdsMultimodalDataset, collate_fn
from tinybyte_mm import ModelConfig, TinyByteModel, _medium_config, _small_config

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRIM_WARN_THRESHOLD = 0.08
TRIM_ABORT_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# LR schedule (unchanged from V3)
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, total_steps, cooldown_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    cooldown_start = total_steps - cooldown_steps
    if step >= cooldown_start:
        frac = (step - cooldown_start) / max(cooldown_steps, 1)
        return max_lr * (1.0 - frac) + min_lr * frac
    progress = (step - warmup_steps) / max(cooldown_start - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# MoD capacity annealing: start fully dense, gradually enable skipping
# ---------------------------------------------------------------------------

def get_mod_capacity(step: int, warmup_steps: int, target_capacity: float) -> float:
    """Linearly anneal MoD capacity from 1.0 (all tokens) to target."""
    if step >= warmup_steps:
        return target_capacity
    return 1.0 - (1.0 - target_capacity) * (step / max(warmup_steps, 1))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    is_distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if is_distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Model config
    if args.small:
        model_cfg = _small_config()
    elif args.medium:
        model_cfg = _medium_config()
    else:
        model_cfg = ModelConfig()

    if args.grad_checkpoint:
        model_cfg.gradient_checkpointing = True
    if args.entropy_patching:
        model_cfg.use_entropy_patching = True
    if args.no_vqvae:
        model_cfg.use_vqvae = False
    if args.no_mod:
        model_cfg.use_mod = False

    model = TinyByteModel(model_cfg).to(device)

    # FSDP wrapping
    if is_distributed and HAS_FSDP:
        mp_policy = None
        if use_bf16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        model = FSDP(model, mixed_precision=mp_policy, cpu_offload=CPUOffload(offload_params=False))
        log.info(f"[Rank {rank}] FSDP enabled")
    elif use_bf16:
        model = model.to(torch.bfloat16)

    if args.compile:
        model = torch.compile(model, fullgraph=False)
        log.info("torch.compile enabled")

    if is_main:
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # [V4] Co-distillation draft model
    draft_model = None
    draft_optimizer = None
    distiller = None
    if args.draft_codistill:
        from draft_model import DraftModel, DraftModelCoDistiller
        draft_model = DraftModel().to(device)
        draft_optimizer = torch.optim.AdamW(
            draft_model.parameters(), lr=args.lr * 3, betas=(0.9, 0.95), weight_decay=0.1
        )
        distiller = DraftModelCoDistiller(
            model, draft_model,
            alpha=args.distill_alpha,
            distill_temperature=args.distill_temperature,
        )
        if is_main:
            log.info(f"Co-distillation enabled: DraftModel {sum(p.numel() for p in draft_model.parameters())/1e6:.1f}M params")

    # Dataset
    data_cfg = DataConfig(
        captions_jsonl=getattr(args, 'captions', './data/captions.jsonl'),
        cache_dir=getattr(args, 'cache_dir', './data/images'),
        max_samples=args.max_samples,
        max_seq_len=model_cfg.max_seq_len,
        wds_path=getattr(args, 'wds_path', None),
    )

    if getattr(args, 'wds_path', None):
        dataset = WdsMultimodalDataset(data_cfg)
        sampler = None
    else:
        dataset = MultimodalByteDataset(data_cfg)
        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset)
            if is_distributed else None
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and not isinstance(dataset, WdsMultimodalDataset)),
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
        fused=(device.type == "cuda"),
    )
    scaler = GradScaler(enabled=(not use_bf16 and device.type == "cuda"))

    steps_per_epoch = len(loader) if hasattr(loader.dataset, "__len__") else args.steps_per_epoch
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = min(2000, total_steps // 20)
    cooldown_steps = min(1000, total_steps // 10)

    # [V4] MoD warmup: ramp over first 10% of total steps
    mod_warmup_steps = total_steps // 10

    if is_main and HAS_WANDB and args.wandb:
        wandb.init(project="tinybyte-v4", config=vars(args))

    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(exist_ok=True)

    global_step = 0
    trim_history = []

    for epoch in range(args.epochs):
        if hasattr(dataset, "_update_curriculum"):
            dataset._update_curriculum(epoch)
        if is_main:
            log.info(
                f"Epoch {epoch}: image_size={data_cfg.current_image_size}, "
                f"jpeg_quality={data_cfg.current_jpeg_quality}"
            )
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)
            is_image  = batch["is_image"].to(device, non_blocking=True)

            # [V4] Update MoD capacity (dense early, then sparse)
            if model_cfg.use_mod and not args.no_mod:
                cap = get_mod_capacity(global_step, mod_warmup_steps, model_cfg.mod_capacity_fraction)
                # Update all MoD routers
                base = model.module if hasattr(model, "module") else model
                for layer in base.layers:
                    if hasattr(layer, "mod_router"):
                        layer.mod_router.capacity_fraction = cap

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))
            )

            with amp_ctx:
                result = model.compute_loss(input_ids, labels, is_image)
                loss = result["loss"] / args.grad_accum

            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            trim_frac = result["trim_frac"]
            trim_history.append(trim_frac)

            if trim_frac > TRIM_ABORT_THRESHOLD:
                log.error(
                    f"ABORT: trim fraction {trim_frac:.1%} exceeds "
                    f"{TRIM_ABORT_THRESHOLD:.0%} at step {global_step}."
                )
                raise RuntimeError("Trim threshold exceeded")
            elif trim_frac > TRIM_WARN_THRESHOLD and is_main:
                log.warning(f"High trim fraction: {trim_frac:.1%} at step {global_step}")

            if (batch_idx + 1) % args.grad_accum == 0:
                lr = get_lr(global_step, warmup_steps, total_steps, cooldown_steps, args.lr, args.lr * 0.1)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if use_bf16:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad()

                # [V4] Co-distillation step
                if distiller is not None:
                    draft_loss_val = distiller.distill_step(
                        input_ids, labels, draft_optimizer,
                        scaler=scaler if not use_bf16 else None
                    )
                else:
                    draft_loss_val = 0.0

                global_step += 1

                if is_main and global_step % args.log_every == 0:
                    metrics = {
                        "loss":           result["loss"].item(),
                        "main_loss":      result["main_loss"],
                        "header_loss":    result["header_loss"],
                        "vq_loss":        result.get("vq_loss", 0.0),
                        "aux_recon_loss": result.get("aux_recon_loss", 0.0),
                        "draft_loss":     draft_loss_val,
                        "trim_frac":      trim_frac,
                        "lr":             lr,
                        "epoch":          epoch,
                        "image_size":     data_cfg.current_image_size,
                    }
                    log.info(
                        f"Step {global_step} | loss={metrics['loss']:.4f} | "
                        f"vq={metrics['vq_loss']:.4f} | trim={trim_frac:.2%} | "
                        f"lr={lr:.2e} | img_size={data_cfg.current_image_size}"
                    )
                    if HAS_WANDB and args.wandb:
                        if global_step % 500 == 0 and trim_history:
                            metrics["trim_histogram"] = wandb.Histogram(trim_history[-500:])
                        wandb.log(metrics, step=global_step)

        # Save checkpoint
        if is_main:
            ckpt_path = checkpoints_dir / f"tinybyte_v4_epoch{epoch}.pt"
            raw_model = model
            if HAS_FSDP and isinstance(model, FSDP):
                with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
                    state = model.state_dict()
            else:
                state = model.state_dict()
            torch.save({"model": state, "config": model_cfg.__dict__, "epoch": epoch}, ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")

            # Save draft model if co-distillation is enabled
            if draft_model is not None:
                draft_dir = checkpoints_dir / f"draft_v4_epoch{epoch}"
                draft_model.save_pretrained(str(draft_dir))

    if is_main:
        log.info("Training complete.")
        if HAS_WANDB and args.wandb:
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TinyByte V4 Trainer")
    p.add_argument("--captions", default="./data/captions.jsonl")
    p.add_argument("--cache-dir", default="./data/images")
    p.add_argument("--wds-path", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    # Model presets
    p.add_argument("--small", action="store_true")
    p.add_argument("--medium", action="store_true")
    # Feature flags
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--entropy-patching", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-vqvae", action="store_true", help="Disable VQ-VAE pathway")
    p.add_argument("--no-mod", action="store_true", help="Disable Mixture-of-Depth")
    # [V4] Co-distillation
    p.add_argument("--draft-codistill", action="store_true", help="Train draft model jointly via co-distillation")
    p.add_argument("--distill-alpha", type=float, default=0.3, help="Weight of hard-label CE in distillation loss")
    p.add_argument("--distill-temperature", type=float, default=2.0, help="Temperature for knowledge distillation")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
