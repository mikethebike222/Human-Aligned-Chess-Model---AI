"""
dataset.py — Loads tokenized .bin files and serves batches to the training loop.

The .bin file is a flat numpy memmap of uint32 tokens — all games concatenated
end-to-end with no separators. To serve batches, we slice fixed-length windows
of 512 tokens at random offsets, then unpack each uint32 into the four separate
signals the model needs:

    input_ids     — lower 14 bits: the token ID (what token is this?)
    labels        — same as input_ids, shifted by 1 (what comes next?)
    time_labels   — upper 16 bits: move time in seconds (normalized)
    value_labels  — bits 14-15: game outcome {-1, 0, +1}
    attention_mask — 1 for real tokens, 0 for padding

The model is trained to predict the NEXT token at every position simultaneously
(standard language model training), so labels = input_ids shifted left by 1.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.chess_tokenizer import normalize_time


class ChessBinDataset(Dataset):
    """
    PyTorch Dataset that reads from a .bin memmap file.

    Instead of loading games individually, we treat the entire token stream as
    one long sequence and sample random fixed-length windows from it. This is
    exactly how GPT-style models are trained on text — the model sees a sliding
    window of context and learns to predict the next token at every step.

    One subtlety: windows can straddle game boundaries (the end of one game
    and start of another). This is fine — the model learns to handle BOS/EOS
    tokens naturally and the attention mask handles padding.
    """

    def __init__(self, bin_path: str, seq_len: int = 512, seed: int = 42):
        """
        Args:
            bin_path:  path to the .bin memmap file (e.g. combined-train.bin)
            seq_len:   number of tokens per training window (default 512)
            seed:      random seed for reproducibility
        """
        self.seq_len   = seq_len
        self.rng       = np.random.default_rng(seed)

        # Memory-map the file — this does NOT load it into RAM.
        # The OS pages in only the chunks we actually access.
        # This is how ALLIE handles 6.6B tokens without running out of memory.
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.n_tokens = len(self.data)

        # Number of complete windows we can draw from the data.
        # Each window needs seq_len + 1 tokens (the +1 is for the label shift).
        self.n_windows = self.n_tokens - seq_len - 1

        print(f"Loaded {bin_path}")
        print(f"  {self.n_tokens:,} tokens  |  {self.n_windows:,} possible windows")

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one training window starting at position idx.

        Unpacks the packed uint32 format into separate tensors:
          - input_ids:     (seq_len,)  token IDs for model input
          - labels:        (seq_len,)  token IDs shifted by 1 (next-token targets)
          - time_labels:   (seq_len,)  normalized move time (-100 if not a move)
          - value_labels:  (seq_len,)  game outcome (-100 if padding)
          - attention_mask:(seq_len,)  1.0 for real tokens, 0.0 for pad
        """
        # Grab seq_len + 1 raw uint32 values
        raw = torch.from_numpy(
            self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        )

        # --- Unpack the three bit fields from each uint32 ---

        # Token IDs: lower 14 bits
        # Values >= vocab_size are ELO tokens (handled in ChessEmbedding)
        all_ids = raw & 0x3FFF                      # (seq_len+1,)

        # Move time: upper 16 bits, stored as (seconds + 1)
        # Value 0 means "not a move token" — we mask those out with -100
        time_encoded = raw >> 16                    # (seq_len+1,)

        # Game outcome: bits 14-15
        # 0 = white wins, 1 = draw, 2 = black wins
        outcome_bits = (raw & 0xFFFF) >> 14         # (seq_len+1,)

        # --- Build model inputs (positions 0..seq_len-1) ---
        input_ids    = all_ids[:-1]                 # (seq_len,)
        time_enc_in  = time_encoded[:-1]            # (seq_len,)
        outcome_in   = outcome_bits[:-1]            # (seq_len,)

        # --- Build targets (positions 1..seq_len) ---
        labels = all_ids[1:].clone()                # (seq_len,)

        # --- Attention mask ---
        # PAD token ID is 2006 in our vocab
        # Tokens beyond ELO range are valid (ELO tokens, not padding)
        PAD_ID = 2006
        attention_mask = (input_ids != PAD_ID).float()

        # --- Time labels ---
        # time_enc = (seconds + 1), so 0 means "not a move" → use -100 (ignored in loss)
        is_move = time_enc_in > 0
        raw_seconds = (time_enc_in - 1).float()
        normalized = torch.tensor(
            [normalize_time(float(s)) for s in raw_seconds],
            dtype=torch.float32,
        )
        time_labels = torch.where(is_move, normalized, torch.full_like(normalized, -100.0))

        # --- Value labels ---
        # outcome_bits: 0=white wins, 1=draw, 2=black wins
        # Map to float: 0→+1.0, 1→0.0, 2→-1.0
        # Mask out positions where attention_mask=0 with -100
        outcome_float = -outcome_in.float() + 1.0  # 0→+1, 1→0, 2→-1
        value_labels  = torch.where(
            attention_mask.bool(),
            outcome_float,
            torch.full_like(outcome_float, -100.0),
        )

        # --- Mask out label positions we don't want to predict ---
        # Don't predict ELO tokens or PAD tokens (they're metadata, not moves)
        VOCAB_SIZE = 2007
        labels[labels >= VOCAB_SIZE] = -100     # ELO tokens
        labels[labels == PAD_ID]     = -100     # PAD tokens
        labels[~attention_mask.bool()] = -100   # PAD positions

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "time_labels":    time_labels,
            "value_labels":   value_labels,
        }


def make_dataloader(
    bin_path: str,
    seq_len: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Convenience function — creates a DataLoader from a .bin file.

    Args:
        bin_path:    path to .bin file
        seq_len:     tokens per sequence window
        batch_size:  sequences per batch
        num_workers: parallel data loading workers
        shuffle:     shuffle windows each epoch
        seed:        random seed
    """
    dataset = ChessBinDataset(bin_path, seq_len=seq_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,    # faster CPU→GPU transfers
        drop_last=True,     # keep batch sizes consistent
    )
