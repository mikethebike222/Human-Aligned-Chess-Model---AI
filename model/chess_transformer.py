"""
chess_transformer.py — Decoder-only transformer for human-aligned chess.

Architecture overview:
  - Base: GPT-2 decoder-only transformer (initialized from pretrained weights)
  - Custom embedding layer that handles both normal tokens AND soft Elo tokens
  - Three output heads on top of the final decoder layer:
      1. policy_head  → probability distribution over next moves
      2. time_head    → predicted seconds a human would spend on this move
      3. value_head   → expected game outcome from current player's perspective

The soft Elo embedding is the key innovation from ALLIE: instead of bucketing
players into discrete Elo ranges, we represent each player's strength as a
continuous interpolation between a learned "weak" embedding (500 Elo) and a
learned "strong" embedding (3000 Elo). This lets the model generalize smoothly
across all skill levels rather than only the 9 discrete Maia models.

Think of it like a dial from 0 to 1: a 1750 Elo player gets a blend that's
roughly 50% of the way between weak and strong.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from model.config import ModelConfig


class ChessEmbedding(nn.Module):
    """
    Custom embedding layer that handles two types of input IDs:

      1. Normal token IDs (0 to vocab_size - 1):
         Standard learned embeddings — one vector per token.

      2. Elo token IDs (>= vocab_size):
         The actual Elo value is recovered as:  elo = token_id - vocab_size
         The embedding is then computed as a linear interpolation between
         two learned embeddings (e_weak and e_strong):

             γ = (elo_max - elo) / (elo_max - elo_min)   # 1.0 at weakest, 0.0 at strongest
             embedding = γ * e_weak + (1 - γ) * e_strong

         This is ALLIE's "soft control token" approach. It solves two problems
         with discrete Elo tokens: (a) data sparsity at exact Elo values, and
         (b) the model not understanding that Elo 1500 and 1505 are nearly identical.
    """

    def __init__(self, vocab_size: int, n_embd: int, elo_min: int, elo_max: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.elo_min = elo_min
        self.elo_max = elo_max

        # Standard token embeddings for move tokens, time controls, specials
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)

        # Two special embeddings for the Elo soft conditioning
        # These are the "anchors" — weak player and strong player embeddings
        self.elo_weak   = nn.Embedding(1, n_embd)   # represents elo_min (500)
        self.elo_strong = nn.Embedding(1, n_embd)   # represents elo_max (3000)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — may contain both normal and Elo IDs

        Returns:
            embeddings: (batch, seq_len, n_embd)
        """
        # Split IDs into normal tokens vs Elo tokens
        is_elo = input_ids >= self.vocab_size

        # --- Normal token embeddings ---
        # Clamp Elo IDs to 0 so the embedding lookup doesn't crash;
        # we'll overwrite those positions below anyway.
        safe_ids = input_ids.clone()
        safe_ids[is_elo] = 0
        embeddings = self.token_embeddings(safe_ids)  # (B, T, C)

        # --- Soft Elo embeddings ---
        if is_elo.any():
            # Recover raw Elo values
            elo_values = (input_ids[is_elo] - self.vocab_size).float()

            # Compute interpolation weight γ: 1.0 = weakest, 0.0 = strongest
            gamma = (self.elo_max - elo_values) / (self.elo_max - self.elo_min)
            gamma = gamma.clamp(0.0, 1.0).unsqueeze(-1)  # (n_elo, 1)

            # Get the two anchor embeddings (both shape: (1, C))
            e_weak   = self.elo_weak(torch.zeros(1, dtype=torch.long, device=input_ids.device))
            e_strong = self.elo_strong(torch.zeros(1, dtype=torch.long, device=input_ids.device))

            # Interpolate: closer to e_weak for low Elo, e_strong for high Elo
            elo_embeddings = gamma * e_weak + (1 - gamma) * e_strong  # (n_elo, C)

            # Write Elo embeddings into the correct positions
            embeddings[is_elo] = elo_embeddings

        return embeddings


class ChessTransformer(nn.Module):
    """
    Decoder-only transformer for chess move prediction.

    Initialized from pretrained GPT-2 weights (all layers except embeddings).
    This weight transfer speeds up training even though chess has nothing to do
    with language — the transformer has already learned useful inductive biases
    about sequential structure.

    Three output heads predict:
      1. Next move (policy)     — cross-entropy loss
      2. Move time in seconds   — MSE loss (normalized)
      3. Game outcome [-1, 1]   — MSE loss (tanh-squashed)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Custom embedding (handles soft Elo tokens) ---
        self.embedding = ChessEmbedding(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            elo_min=config.elo_min,
            elo_max=config.elo_max,
        )

        # --- GPT-2 transformer backbone ---
        # We use GPT-2's transformer blocks directly but replace its embedding
        # and LM head with our custom versions.
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,   # unused (we handle embeddings ourselves)
            n_positions=config.max_seq_len,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            resid_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            embd_pdrop=0.0,                 # embedding dropout disabled (we handle it)
        )
        self.transformer = GPT2Model(gpt2_config)

        # --- Three output heads ---
        # All are linear layers applied to the final hidden state at each position.

        # 1. Policy head: predicts which move comes next
        #    Output shape: (batch, seq_len, vocab_size)
        #    Used with cross-entropy loss against the actual next token
        self.policy_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 2. Time head: predicts how long a human would think (in normalized seconds)
        #    Output shape: (batch, seq_len, 1)
        #    Used with MSE loss against actual move times (after normalization)
        self.time_head = nn.Linear(config.n_embd, 1)

        # 3. Value head: predicts game outcome from current position
        #    Output shape: (batch, seq_len, 1), squashed to [-1, 1] by tanh
        #    -1 = black wins, 0 = draw, +1 = white wins
        #    Used with MSE loss against actual game outcomes
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, 1),
            nn.Tanh(),
        )

        # --- Load pretrained GPT-2 weights ---
        self._init_from_gpt2(config.pretrained_gpt2)

    def forward(
        self,
        input_ids: torch.Tensor,            # (batch, seq_len)
        attention_mask: torch.Tensor,       # (batch, seq_len) — 1 for real tokens, 0 for pad
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding → transformer → three heads.

        Returns a dict with keys:
            "policy_logits": (batch, seq_len, vocab_size)  — raw (unnormalized) move scores
            "time_pred":     (batch, seq_len)               — predicted move time (normalized)
            "value_pred":    (batch, seq_len)               — predicted outcome in [-1, 1]
        """
        # 1. Embed input tokens (handles both normal tokens and Elo soft tokens)
        hidden = self.embedding(input_ids)          # (B, T, C)

        # 2. Pass through transformer decoder blocks
        #    GPT2Model expects inputs_embeds when we handle embedding ourselves
        transformer_out = self.transformer(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_out.last_hidden_state  # (B, T, C)

        # 3. Apply all three output heads
        policy_logits = self.policy_head(hidden_states)             # (B, T, vocab_size)
        time_pred     = self.time_head(hidden_states).squeeze(-1)   # (B, T)
        value_pred    = self.value_head(hidden_states).squeeze(-1)  # (B, T)

        return {
            "policy_logits": policy_logits,
            "time_pred":     time_pred,
            "value_pred":    value_pred,
        }

    def _init_from_gpt2(self, model_id: str) -> None:
        """
        Load pretrained GPT-2 transformer weights, skipping the embedding
        and LM head (since our vocab has nothing to do with text).

        Why does this help? GPT-2 has already learned that transformers should
        pay attention to nearby context, that some positions matter more than
        others, etc. These structural biases transfer even to non-language tasks.
        """
        from transformers import GPT2Model as HF_GPT2
        print(f"Loading pretrained weights from {model_id}...")

        pretrained = HF_GPT2.from_pretrained(model_id)

        # Copy only the transformer block weights — NOT the embedding or LM head.
        # Our embedding handles chess tokens + Elo; GPT-2's handles text tokens.
        pretrained_state = pretrained.state_dict()
        our_state        = self.transformer.state_dict()

        # Only copy keys that exist in both and have the same shape
        n_copied = 0
        for key in our_state:
            if key in pretrained_state and our_state[key].shape == pretrained_state[key].shape:
                our_state[key] = pretrained_state[key]
                n_copied += 1

        self.transformer.load_state_dict(our_state)
        print(f"  Copied {n_copied} weight tensors from {model_id}.")
        print(f"  Embedding and output heads are randomly initialized.")

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
