"""
train.py — Training loop for the chess transformer.

Trains the model to jointly minimize three losses:
  1. Cross-entropy on next move prediction  (policy)
  2. MSE on move time prediction            (time)
  3. MSE on game outcome prediction         (value)

Total loss = CE + MSE_time + MSE_value  (equal weighting, same as ALLIE)

Usage:
    # Validation run — SmallConfig on EC2, confirm pipeline works
    python train.py --config small --train_bin ~/data/combined-train.bin --steps 1000

    # Full training run — MediumConfig on Northeastern cluster
    python train.py --config medium --train_bin ~/data/combined-train.bin --steps 2000000
"""

import argparse
import math
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.chess_transformer import ChessTransformer
from model.config import SmallConfig, MediumConfig
from training.dataset import make_dataloader


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict,
    labels: torch.Tensor,
    time_labels: torch.Tensor,
    value_labels: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the three-part loss.

    Labels use -100 as the ignore index (standard PyTorch convention) —
    positions with -100 are excluded from the loss calculation entirely.
    This handles ELO tokens, PAD tokens, and non-move positions cleanly.

    Args:
        outputs:      dict from model.forward() with policy_logits, time_pred, value_pred
        labels:       (B, T) next token IDs, -100 where we don't predict
        time_labels:  (B, T) normalized move times, -100 where not a move
        value_labels: (B, T) game outcomes in [-1, 1], -100 for padding

    Returns:
        total_loss: scalar tensor (CE + MSE_time + MSE_value)
        loss_parts: dict with individual loss values for logging
    """
    B, T, V = outputs["policy_logits"].shape

    # 1. Policy loss — cross-entropy over vocab at each position
    # Reshape to (B*T, V) for the loss function
    policy_logits = outputs["policy_logits"].reshape(B * T, V)
    policy_labels = labels.reshape(B * T)
    policy_loss = nn.functional.cross_entropy(policy_logits, policy_labels, ignore_index=-100)

    # 2. Time loss — MSE on move time prediction
    # Only compute loss where time_labels != -100 (i.e., actual move positions)
    time_pred = outputs["time_pred"]                          # (B, T)
    time_mask = time_labels != -100
    if time_mask.any():
        time_loss = nn.functional.mse_loss(
            time_pred[time_mask],
            time_labels[time_mask],
        )
    else:
        time_loss = torch.tensor(0.0, device=time_pred.device)

    # 3. Value loss — MSE on game outcome prediction
    # Only compute loss where value_labels != -100
    value_pred = outputs["value_pred"]                        # (B, T)
    value_mask = value_labels != -100
    if value_mask.any():
        value_loss = nn.functional.mse_loss(
            value_pred[value_mask],
            value_labels[value_mask],
        )
    else:
        value_loss = torch.tensor(0.0, device=value_pred.device)

    total_loss = policy_loss + time_loss + value_loss

    return total_loss, {
        "loss/total":  total_loss.item(),
        "loss/policy": policy_loss.item(),
        "loss/time":   time_loss.item(),
        "loss/value":  value_loss.item(),
    }


# ---------------------------------------------------------------------------
# Learning rate scheduler — cosine decay with linear warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, max_steps: int, lr_max: float, lr_min: float, warmup_steps: int) -> float:
    """
    Linear warmup for the first `warmup_steps`, then cosine decay to `lr_min`.

    This is the standard schedule used in GPT training. Warmup prevents large
    gradient updates at the start when the model weights are random.
    """
    if step < warmup_steps:
        # Linear warmup: 0 → lr_max over warmup_steps
        return lr_max * step / warmup_steps
    if step > max_steps:
        return lr_min
    # Cosine decay: lr_max → lr_min over the remaining steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, loss, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint-{step:07d}.pt")
    torch.save({
        "step":       step,
        "loss":       loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']} (loss={ckpt['loss']:.4f})")
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Config ---
    config = SmallConfig() if args.config == "small" else MediumConfig()
    if args.steps:
        config.max_steps = args.steps
    print(f"Config: {config.__class__.__name__}  ({config.n_layer}L {config.n_embd}D)")

    # --- Model ---
    model = ChessTransformer(config).to(device)
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

    # Use bfloat16 on GPU for faster training (same as ALLIE)
    # bfloat16 has the same exponent range as float32 but less precision —
    # it's more numerically stable than float16 for training
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype == torch.bfloat16:
        model = model.to(dtype)
        print("Precision: bfloat16")

    # --- Optimizer ---
    # AdamW with weight decay on weights but NOT on biases/LayerNorm params
    # (standard best practice for transformer training)
    decay_params  = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))

    # --- Resume from checkpoint if provided ---
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)

    # --- Data ---
    # batch_size = how many sequences fit in our token budget per step
    batch_size = max(1, config.batch_tokens // config.max_seq_len)
    loader = make_dataloader(
        args.train_bin,
        seq_len=config.max_seq_len,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    data_iter = iter(loader)
    print(f"Batch size: {batch_size} sequences × {config.max_seq_len} tokens = "
          f"{batch_size * config.max_seq_len:,} tokens/step")

    # --- Training loop ---
    model.train()
    step = start_step
    t0   = time.time()

    while step < config.max_steps:
        # Refresh data iterator at end of epoch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move batch to device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        time_labels    = batch["time_labels"].to(device)
        value_labels   = batch["value_labels"].to(device)

        # Cast to bfloat16 if needed
        if dtype == torch.bfloat16:
            attention_mask = attention_mask.to(dtype)
            time_labels    = time_labels.to(dtype)
            value_labels   = value_labels.to(dtype)

        # Update learning rate (cosine schedule)
        lr = get_lr(step, config.max_steps, config.learning_rate, config.lr_min, config.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss, loss_parts = compute_loss(outputs, labels, time_labels, value_labels)

        # Backward pass
        loss.backward()

        # Clip gradients — prevents exploding gradients early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        step += 1

        # --- Logging ---
        if step % args.log_every == 0:
            dt = time.time() - t0
            tokens_per_sec = args.log_every * batch_size * config.max_seq_len / dt
            print(
                f"step {step:7d} | "
                f"loss {loss_parts['loss/total']:.4f} "
                f"(policy={loss_parts['loss/policy']:.4f} "
                f"time={loss_parts['loss/time']:.4f} "
                f"value={loss_parts['loss/value']:.4f}) | "
                f"lr={lr:.2e} | "
                f"{tokens_per_sec:,.0f} tok/s"
            )
            t0 = time.time()

        # --- Checkpoint ---
        if step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, loss_parts["loss/total"], args.out_dir)

    # Save final checkpoint
    save_checkpoint(model, optimizer, step, loss_parts["loss/total"], args.out_dir)
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the chess transformer")
    parser.add_argument("--config",      choices=["small", "medium"], default="small")
    parser.add_argument("--train_bin",   required=True,  help="Path to combined-train.bin")
    parser.add_argument("--out_dir",     default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--resume",      default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--steps",       type=int, default=None, help="Override max training steps")
    parser.add_argument("--log_every",   type=int, default=100,  help="Log every N steps")
    parser.add_argument("--save_every",  type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--num_workers", type=int, default=4,    help="DataLoader worker processes")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
