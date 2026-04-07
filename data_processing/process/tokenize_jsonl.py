"""
tokenize_jsonl.py — Convert JSONL game files to a uint32 memmap .bin file.

Reads one or more JSONL files (output of pgn_parser.py), tokenizes each game
using ChessTokenizer, and writes a flat uint32 numpy memmap — same format as
ALLIE's 2022.bin so both can be concatenated directly.

Usage:
    python3 data_processing/process/tokenize_jsonl.py \
        --input_files ~/data/lichess-2026-rapid/2026-01.jsonl \
                      ~/data/lichess-2026-rapid/2026-02.jsonl \
                      ~/data/lichess-2026-rapid/2026-03.jsonl \
        --output ~/data/lichess-2026-rapid/2026-rapid.bin \
        --mode rapid

    # Check how many tokens were written:
    python3 -c "import numpy as np; d=np.memmap('2026-rapid.bin', dtype=np.uint32, mode='r'); print(f'{len(d):,} tokens')"
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tokenizer.chess_tokenizer import ChessTokenizer


def count_lines(path: str) -> int:
    """Fast line count via iteration."""
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def tokenize_jsonl_files(input_files: List[str], output_path: str, mode: str):
    tokenizer = ChessTokenizer()

    # --- First pass: count total tokens so we can pre-allocate the memmap ---
    print("Counting tokens (first pass)...")
    total_tokens = 0
    game_count = 0

    for path in input_files:
        print(f"  Scanning {path} ...")
        with open(path) as f:
            for line in tqdm(f, desc=os.path.basename(path), unit=" games"):
                game = json.loads(line)
                tokens = tokenizer.tokenize(
                    game,
                    add_elo=True,
                    add_time_control=True,
                    add_termination=True,
                )
                total_tokens += len(tokens)
                game_count += 1

    print(f"  {game_count:,} games → {total_tokens:,} tokens")

    # --- Allocate output memmap ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))

    # --- Second pass: write tokens ---
    print("Writing tokens (second pass)...")
    offset = 0

    for path in input_files:
        print(f"  Tokenizing {path} ...")
        with open(path) as f:
            for line in tqdm(f, desc=os.path.basename(path), unit=" games"):
                game = json.loads(line)
                tokens = tokenizer.tokenize(
                    game,
                    add_elo=True,
                    add_time_control=True,
                    add_termination=True,
                )
                n = len(tokens)
                out[offset : offset + n] = tokens
                offset += n

    out.flush()
    print(f"\nWrote {offset:,} tokens to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Tokenize JSONL game files → uint32 .bin")
    parser.add_argument(
        "--input_files", nargs="+", required=True, help="JSONL files to tokenize"
    )
    parser.add_argument("--output", required=True, help="Output .bin file path")
    parser.add_argument(
        "--mode",
        choices=["rapid", "blitz"],
        default="rapid",
        help="Game mode (for logging only)",
    )
    args = parser.parse_args()

    tokenize_jsonl_files(args.input_files, args.output, args.mode)


if __name__ == "__main__":
    main()
