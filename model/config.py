"""
config.py — Hyperparameters for the chess transformer model.

Two configs are defined:
  - SmallConfig:  GPT-2 small (124M params) — use this for local testing and
                  pipeline validation on EC2. Trains fast, catches bugs cheaply.
  - MediumConfig: GPT-2 medium (355M params) — the full ALLIE-scale model.
                  Use this on Northeastern's GPU cluster for the real training run.

You should validate the entire pipeline with SmallConfig before committing
to the full MediumConfig training run.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # -------------------------------------------------------------------
    # Vocabulary
    # -------------------------------------------------------------------
    vocab_size: int = 2007          # 1968 moves + 24 blitz TC + 10 rapid TC
                                    # + 1 termination + 4 special tokens
    elo_min: int = 500              # lowest Elo we expect to condition on
    elo_max: int = 3000             # highest Elo

    # -------------------------------------------------------------------
    # Transformer architecture
    # -------------------------------------------------------------------
    n_layer: int = 12               # number of transformer decoder blocks
    n_head: int = 12                # number of attention heads per block
    n_embd: int = 768               # embedding dimension (d_model)
    max_seq_len: int = 512          # maximum token sequence length

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    dropout: float = 0.1            # dropout rate (applied to attention + MLP)
    learning_rate: float = 6e-4     # initial LR (decays to lr_min via cosine)
    lr_min: float = 1e-5            # final LR after cosine decay
    batch_tokens: int = 131_072     # global batch size in tokens (packed sequences)
    max_steps: int = 2_000_000      # total training steps (~40 epochs over 6.6B tokens)
    warmup_steps: int = 2_000       # linear LR warmup

    # -------------------------------------------------------------------
    # GPT-2 weight initialization
    # -------------------------------------------------------------------
    # We initialize transformer weights (attention, MLP, LayerNorm) from
    # pretrained GPT-2 to benefit from its learned structure. Embeddings
    # are trained from scratch since our vocab has nothing to do with text.
    pretrained_gpt2: str = "gpt2"   # HuggingFace model ID for weight init


@dataclass
class SmallConfig(ModelConfig):
    """
    GPT-2 small (124M params) — for pipeline validation.

    Use this to verify the full train → eval loop works correctly
    before running the expensive MediumConfig job on the cluster.
    Everything else (dataset, tokenizer, loss) is identical.
    """
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    pretrained_gpt2: str = "gpt2"           # GPT-2 small weights

    # Shorter training for validation runs
    max_steps: int = 1_000                  # just enough to confirm loss decreases
    batch_tokens: int = 4_096               # smaller batch for local GPU/CPU testing


@dataclass
class MediumConfig(ModelConfig):
    """
    GPT-2 medium (355M params) — the full ALLIE-scale model.

    Matches the architecture described in the paper. Train this on
    Northeastern's GPU cluster with 8x A6000s.
    """
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    pretrained_gpt2: str = "gpt2-medium"    # GPT-2 medium weights

    # Full training run
    max_steps: int = 2_000_000
    batch_tokens: int = 131_072
