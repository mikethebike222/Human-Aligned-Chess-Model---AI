"""
eval.py — Evaluate the chess transformer against Maia baselines.

Primary metric: move-matching accuracy — how often does the model's top-1
predicted move match the move a human actually played?

Evaluated per 100-Elo bin across the full skill spectrum, following the
same methodology as the ALLIE paper (Table 3 / Figure 2).

Two positions are excluded from evaluation (same as ALLIE):
  - First 5 moves of each game (opening book — too many humans play the same moves)
  - Moves made with < 30s remaining on clock (time pressure = semi-random moves)

Target: beat Maia*'s overall accuracy of 51.6% (ALLIE paper Table 3).

Usage:
    python eval.py \\
        --checkpoint checkpoints/checkpoint-2000000.pt \\
        --test_dir   ~/data/lichess-2022-blitz-sampled \\
        --config     medium
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model.chess_transformer import ChessTransformer
from model.config import SmallConfig, MediumConfig
from tokenizer.chess_tokenizer import ChessTokenizer
from data_processing.files.chess_moves import CHESS_MOVES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elo_bin(elo: int, bin_size: int = 100) -> str:
    """Round an Elo to the nearest bin. e.g. 1543 → '1500'."""
    return str((elo // bin_size) * bin_size)


def load_test_games(test_dir: str) -> List[dict]:
    """Load all *test.jsonl files from the test directory."""
    games = []
    for fname in sorted(os.listdir(test_dir)):
        if fname.endswith("test.jsonl"):
            with open(os.path.join(test_dir, fname)) as f:
                for line in f:
                    games.append(json.loads(line))
    print(f"Loaded {len(games):,} test games from {test_dir}")
    return games


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, games, device, config):
    """
    Run move-matching accuracy evaluation on a list of test games.

    For each game, we feed the move history token-by-token and ask the model
    to predict the next move. We count how often its top-1 prediction matches
    the actual human move.

    Returns:
        overall_accuracy: float (0–1)
        per_bin_accuracy: dict mapping Elo bin string → accuracy float
    """
    model.eval()

    # We'll track correct and total predictions per Elo bin
    bin_correct = defaultdict(int)
    bin_total   = defaultdict(int)

    MOVE_ID_SET = set(range(len(CHESS_MOVES)))  # valid move token IDs (0–1967)

    for game_idx, game in enumerate(games):
        if game_idx % 1000 == 0:
            print(f"  Evaluating game {game_idx:,} / {len(games):,} ...", end="\r")

        moves      = game["moves-uci"].split()
        times      = game["moves-seconds"]
        white_elo  = int(game["white-elo"])
        black_elo  = int(game["black-elo"])
        time_ctrl  = game.get("time-control", "")

        # Parse time control for clock tracking
        try:
            base_time, increment = map(int, time_ctrl.split("+"))
        except Exception:
            continue

        # Build the token prefix up to each position and predict next move
        # We start from move index 5 (skip first 5 moves per ALLIE eval protocol)
        clock = [base_time, base_time]

        for move_idx in range(len(moves)):
            # Track remaining clock time
            turn = move_idx % 2
            if move_idx < len(times):
                clock[turn] = max(0, clock[turn] - times[move_idx] + increment)

            # Skip first 5 moves (opening book)
            if move_idx < 5:
                continue

            # Skip moves made under time pressure (< 30s remaining)
            if clock[turn] < 30:
                continue

            # The move we're trying to predict is moves[move_idx]
            actual_move = moves[move_idx]
            actual_move_id = tokenizer.get_token_id(actual_move)
            if actual_move_id == tokenizer.unk_id:
                continue  # move not in vocab, skip

            # Build context: all moves BEFORE this position
            context_game = {
                "moves-uci":     " ".join(moves[:move_idx]),
                "moves-seconds": times[:move_idx],
                "white-elo":     str(white_elo),
                "black-elo":     str(black_elo),
                "time-control":  time_ctrl,
                "result":        game.get("result", "1/2-1/2"),
                "termination":   "",
            }

            # Tokenize context (no termination/EOS — game isn't over yet)
            token_array = tokenizer.tokenize(
                context_game,
                add_elo=True,
                add_time_control=True,
                add_termination=False,
            )

            # Truncate to max_seq_len from the RIGHT (keep most recent context)
            max_len = config.max_seq_len
            if len(token_array) > max_len:
                token_array = token_array[-max_len:]

            # Unpack token IDs (lower 14 bits) — strip time/outcome metadata
            input_ids = torch.tensor(
                (token_array & 0x3FFF).astype(int),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)  # (1, T)

            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

            # Forward pass — we only need the prediction at the LAST position
            outputs = model(input_ids, attention_mask)
            last_logits = outputs["policy_logits"][0, -1, :]  # (vocab_size,)

            # Mask out non-move tokens — we only want move predictions
            # Set logits of special tokens, time controls, ELO tokens to -inf
            move_mask = torch.full((config.vocab_size,), float("-inf"), device=device)
            move_mask[:len(CHESS_MOVES)] = 0.0  # allow move tokens
            last_logits = last_logits + move_mask

            # Top-1 predicted move
            predicted_id = int(last_logits.argmax())

            # Score it — use average of white and black Elo for the bin
            avg_elo = (white_elo + black_elo) // 2
            bin_key = elo_bin(avg_elo)

            bin_total[bin_key]   += 1
            bin_correct[bin_key] += int(predicted_id == actual_move_id)

    print()  # clear the \r line

    # Compute per-bin and overall accuracy
    per_bin = {}
    total_correct = 0
    total_count   = 0
    for bin_key in sorted(bin_correct.keys(), key=lambda x: int(x)):
        n     = bin_total[bin_key]
        c     = bin_correct[bin_key]
        acc   = c / n if n > 0 else 0.0
        per_bin[bin_key] = acc
        total_correct += c
        total_count   += n

    overall = total_correct / total_count if total_count > 0 else 0.0
    return overall, per_bin


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results(overall: float, per_bin: dict):
    """Print a formatted results table."""
    print()
    print("=" * 55)
    print(f"  Overall move-matching accuracy:  {overall * 100:.2f}%")
    print(f"  ALLIE paper baseline (Maia*):     51.60%")
    print(f"  ALLIE paper (ALLIE-POLICY):        55.70%")
    print("=" * 55)
    print(f"  {'Elo bin':>10}  {'Accuracy':>10}  {'vs Maia*':>10}")
    print("-" * 55)

    # Approximate Maia* accuracy per bin from the ALLIE paper (Figure 2)
    # These are rough values read from the figure for reference
    maia_approx = {
        "600": 0.42, "700": 0.43, "800": 0.44, "900": 0.46,
        "1000": 0.47, "1100": 0.48, "1200": 0.49, "1300": 0.50,
        "1400": 0.51, "1500": 0.52, "1600": 0.52, "1700": 0.53,
        "1800": 0.53, "1900": 0.53, "2000": 0.52, "2100": 0.52,
        "2200": 0.51, "2300": 0.50, "2400": 0.49, "2500": 0.48,
    }

    for bin_key in sorted(per_bin.keys(), key=lambda x: int(x)):
        acc   = per_bin[bin_key]
        maia  = maia_approx.get(bin_key, None)
        delta = f"+{(acc - maia) * 100:.1f}%" if maia else "  n/a"
        print(f"  {bin_key + '-' + str(int(bin_key)+100):>10}  {acc * 100:>9.2f}%  {delta:>10}")

    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate chess transformer vs Maia")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--test_dir",   required=True, help="Directory with *test.jsonl files")
    parser.add_argument("--config",     choices=["small", "medium"], default="medium")
    parser.add_argument("--max_games",  type=int, default=None, help="Limit number of test games")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config + model
    config = SmallConfig() if args.config == "small" else MediumConfig()
    model  = ChessTransformer(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # Load tokenizer
    tokenizer = ChessTokenizer()

    # Load test games
    games = load_test_games(args.test_dir)
    if args.max_games:
        games = games[:args.max_games]

    # Run evaluation
    print(f"Evaluating {len(games):,} games...")
    overall, per_bin = evaluate(model, tokenizer, games, device, config)

    # Print results
    print_results(overall, per_bin)


if __name__ == "__main__":
    main()
