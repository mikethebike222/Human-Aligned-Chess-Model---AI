"""
concat_bins.py — Concatenate two (or more) uint32 memmap .bin files.

Writes a new flat uint32 memmap containing all tokens from each input file
in order. Also recomputes TIME_MEAN and TIME_STDEV from the combined dataset
so the tokenizer normalization constants stay accurate.

Usage:
    python3 data_processing/process/concat_bins.py \
        --inputs ~/data/lichess-2022-blitz-train/2022-patched.bin \
                 ~/data/lichess-2026-rapid/2026-rapid.bin \
        --output ~/data/combined-train.bin \
        --recompute_time_stats

The TIME_MEAN / TIME_STDEV printed at the end should be copied into
tokenizer/chess_tokenizer.py if they differ significantly from the current
defaults (TIME_MEAN=4.64, TIME_STDEV=6.17).
"""

import argparse
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def concat_bins(input_paths: List[str], output_path: str, recompute_stats: bool):
    # --- Measure total size ---
    total = 0
    sizes = []
    for p in input_paths:
        d = np.memmap(p, dtype=np.uint32, mode="r")
        sizes.append(len(d))
        total += len(d)
        print(f"  {p}: {len(d):,} tokens  ({os.path.getsize(p)/1e9:.2f} GB)")

    print(f"\nTotal: {total:,} tokens  ({total * 4 / 1e9:.2f} GB)")

    # --- Allocate output ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(total,))

    # --- Copy in chunks ---
    CHUNK = 50_000_000  # 50M tokens per chunk (~200MB)
    offset = 0
    for p in input_paths:
        src = np.memmap(p, dtype=np.uint32, mode="r")
        n = len(src)
        for start in tqdm(range(0, n, CHUNK), desc=os.path.basename(p)):
            end = min(start + CHUNK, n)
            out[offset : offset + (end - start)] = src[start:end]
            offset += end - start
        del src

    out.flush()
    print(f"\nWrote {offset:,} tokens to {output_path}")

    # --- Optionally recompute time normalization stats ---
    if recompute_stats:
        print("\nRecomputing TIME_MEAN / TIME_STDEV ...")
        # Move tokens have time > 0 in upper 16 bits: (packed >> 16) - 1 >= 0
        # i.e., (packed >> 16) >= 1
        data = np.memmap(output_path, dtype=np.uint32, mode="r")

        # Sample up to 100M tokens to keep it fast
        sample_size = min(100_000_000, len(data))
        idx = np.random.choice(len(data), sample_size, replace=False)
        sample = data[idx]

        raw_times = (sample >> 16).astype(np.int32) - 1  # -1 means not a move
        move_mask = raw_times >= 0
        move_times = raw_times[move_mask].astype(np.float64)

        mean = float(move_times.mean())
        std  = float(move_times.std())
        print(f"\n  TIME_MEAN  = {mean:.5f}")
        print(f"  TIME_STDEV = {std:.5f}")
        print(f"  (computed from {move_mask.sum():,} move tokens in sample)")
        print(f"\n  If these differ from the current defaults (4.64001, 6.16533),")
        print(f"  update TIME_MEAN and TIME_STDEV in tokenizer/chess_tokenizer.py")


def main():
    parser = argparse.ArgumentParser(description="Concatenate uint32 .bin files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .bin files (in order)")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument(
        "--recompute_time_stats",
        action="store_true",
        help="Recompute TIME_MEAN / TIME_STDEV from combined dataset",
    )
    args = parser.parse_args()

    concat_bins(args.inputs, args.output, args.recompute_time_stats)


if __name__ == "__main__":
    main()
