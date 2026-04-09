"""
eval.py — Evaluate the chess transformer against Maia baselines.

Primary metric: move-matching accuracy — how often does the model's top-1
predicted move match the move a human actually played?

Evaluated per 100-Elo bin across the full skill spectrum, following the
same methodology as the ALLIE paper (Table 3 / Figure 2).

Also evaluates:
  - Special move accuracy: castling, pawn promotions
  - Resignation prediction: does the model predict the termination token
    at the end of resigned/checkmated games?

Two positions are excluded from evaluation (same as ALLIE):
  - First 5 moves of each game (opening book — too many humans play the same moves)
  - Moves made with < 30s remaining on clock (time pressure = semi-random moves)

Target: beat Maia*'s overall accuracy of 51.6% (ALLIE paper Table 3).

Usage:
    python eval.py \\
        --checkpoint checkpoints/checkpoint-2000000.pt \\
        --test_dir   ~/data/test \\
        --config     small
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
# Move classification helpers
# ---------------------------------------------------------------------------

# The 4 castling moves in UCI notation
CASTLING_MOVES = {"e1g1", "e1c1", "e8g8", "e8c8"}

# Promotion piece suffixes
PROMOTION_PIECES = {"q", "r", "b", "n"}


def classify_move(uci: str) -> str:
    """
    Classify a UCI move string into a category.

    Returns one of: "castling", "promotion", "regular"
    """
    if uci in CASTLING_MOVES:
        return "castling"
    if len(uci) == 5 and uci[-1] in PROMOTION_PIECES:
        return "promotion"
    return "regular"


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

    Returns a results dict containing:
        overall_accuracy:     float
        per_bin_accuracy:     dict bin_str → float
        special_move_stats:   dict with castling / promotion breakdowns
        resignation_accuracy: float
    """
    model.eval()

    MOVE_ID_SET  = set(range(len(CHESS_MOVES)))
    TERMINATION_ID = tokenizer.termination_id

    # --- Per-bin tracking ---
    bin_correct = defaultdict(int)
    bin_total   = defaultdict(int)

    # --- Special move tracking ---
    special_correct = defaultdict(int)  # key: "castling" / "promotion" / "regular"
    special_total   = defaultdict(int)

    # --- Resignation tracking ---
    resign_correct = 0
    resign_total   = 0

    for game_idx, game in enumerate(games):
        if game_idx % 500 == 0:
            print(f"  Evaluating game {game_idx:,} / {len(games):,} ...", end="\r")

        moves     = game["moves-uci"].split()
        times     = game["moves-seconds"]
        white_elo = int(game["white-elo"])
        black_elo = int(game["black-elo"])
        time_ctrl = game.get("time-control", "")
        result    = game.get("result", "")
        term      = game.get("termination", "")

        try:
            base_time, increment = map(int, time_ctrl.split("+"))
        except Exception:
            continue

        clock = [base_time, base_time]
        avg_elo = (white_elo + black_elo) // 2
        bin_key = elo_bin(avg_elo)

        # -----------------------------------------------------------------
        # Move-matching loop
        # -----------------------------------------------------------------
        for move_idx in range(len(moves)):
            turn = move_idx % 2
            if move_idx < len(times):
                clock[turn] = max(0, clock[turn] - times[move_idx] + increment)

            if move_idx < 5:
                continue
            if clock[turn] < 30:
                continue

            actual_move    = moves[move_idx]
            actual_move_id = tokenizer.get_token_id(actual_move)
            if actual_move_id == tokenizer.unk_id:
                continue

            # Build context up to this position
            context_game = {
                "moves-uci":     " ".join(moves[:move_idx]),
                "moves-seconds": times[:move_idx],
                "white-elo":     str(white_elo),
                "black-elo":     str(black_elo),
                "time-control":  time_ctrl,
                "result":        result,
                "termination":   "",
            }

            token_array = tokenizer.tokenize(
                context_game,
                add_elo=True,
                add_time_control=True,
                add_termination=False,
            )

            max_len = config.max_seq_len
            if len(token_array) > max_len:
                token_array = token_array[-max_len:]

            input_ids = torch.tensor(
                (token_array & 0x3FFF).astype(int),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

            outputs     = model(input_ids, attention_mask)
            last_logits = outputs["policy_logits"][0, -1, :]

            # Mask to move tokens only
            move_mask = torch.full((config.vocab_size,), float("-inf"), device=device)
            move_mask[:len(CHESS_MOVES)] = 0.0
            last_logits = last_logits + move_mask

            predicted_id = int(last_logits.argmax())
            is_correct   = int(predicted_id == actual_move_id)

            # Per-bin accuracy
            bin_total[bin_key]   += 1
            bin_correct[bin_key] += is_correct

            # Special move accuracy
            category = classify_move(actual_move)
            special_total[category]   += 1
            special_correct[category] += is_correct

        # -----------------------------------------------------------------
        # Resignation / checkmate prediction
        # Only evaluate games that ended in resignation or checkmate
        # -----------------------------------------------------------------
        is_terminal = (
            term.lower() in ("normal",)
            and result in ("1-0", "0-1")
            and len(moves) >= 5
        )
        if is_terminal:
            # Feed the full game and check if model predicts termination token
            full_game = {
                "moves-uci":     " ".join(moves),
                "moves-seconds": times,
                "white-elo":     str(white_elo),
                "black-elo":     str(black_elo),
                "time-control":  time_ctrl,
                "result":        result,
                "termination":   "",
            }

            token_array = tokenizer.tokenize(
                full_game,
                add_elo=True,
                add_time_control=True,
                add_termination=False,
            )

            max_len = config.max_seq_len
            if len(token_array) > max_len:
                token_array = token_array[-max_len:]

            input_ids = torch.tensor(
                (token_array & 0x3FFF).astype(int),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

            outputs     = model(input_ids, attention_mask)
            last_logits = outputs["policy_logits"][0, -1, :]

            # Here we do NOT mask to moves only — we want to see if the model
            # predicts the termination token over all vocab tokens
            predicted_id = int(last_logits.argmax())
            resign_total   += 1
            resign_correct += int(predicted_id == TERMINATION_ID)

    print()

    # Compute per-bin accuracy
    per_bin = {}
    total_correct = 0
    total_count   = 0
    for bin_key in sorted(bin_correct.keys(), key=lambda x: int(x)):
        n = bin_total[bin_key]
        c = bin_correct[bin_key]
        per_bin[bin_key] = c / n if n > 0 else 0.0
        total_correct += c
        total_count   += n

    overall = total_correct / total_count if total_count > 0 else 0.0

    # Special move stats
    special_stats = {}
    for cat in ("castling", "promotion", "regular"):
        n = special_total[cat]
        c = special_correct[cat]
        special_stats[cat] = {"correct": c, "total": n, "accuracy": c / n if n > 0 else 0.0}

    resign_acc = resign_correct / resign_total if resign_total > 0 else 0.0

    return {
        "overall":     overall,
        "per_bin":     per_bin,
        "special":     special_stats,
        "resignation": {"accuracy": resign_acc, "correct": resign_correct, "total": resign_total},
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results(results: dict):
    """Print a formatted results table."""
    overall  = results["overall"]
    per_bin  = results["per_bin"]
    special  = results["special"]
    resign   = results["resignation"]

    # --- Overall + Elo bin table ---
    print()
    print("=" * 55)
    print(f"  Overall move-matching accuracy:  {overall * 100:.2f}%")
    print(f"  ALLIE paper baseline (Maia*):     51.60%")
    print(f"  ALLIE paper (ALLIE-POLICY):        55.70%")
    print("=" * 55)
    print(f"  {'Elo bin':>10}  {'Accuracy':>10}  {'vs Maia*':>10}")
    print("-" * 55)

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

    # --- Special move breakdown ---
    print()
    print("=" * 55)
    print("  SPECIAL MOVE ACCURACY")
    print("=" * 55)
    print(f"  {'Category':>12}  {'Accuracy':>10}  {'Correct':>8}  {'Total':>8}")
    print("-" * 55)
    for cat in ("regular", "castling", "promotion"):
        s   = special[cat]
        acc = s["accuracy"] * 100
        print(f"  {cat:>12}  {acc:>9.2f}%  {s['correct']:>8,}  {s['total']:>8,}")
    print("=" * 55)

    # --- Resignation prediction ---
    print()
    print("=" * 55)
    print("  RESIGNATION / CHECKMATE PREDICTION")
    print("=" * 55)
    racc = resign["accuracy"] * 100
    print(f"  Accuracy:  {racc:.2f}%  ({resign['correct']:,} / {resign['total']:,} terminal positions)")
    print(f"  (% of game-ending positions where model predicts <RESIGNED-OR-CHECKMATED>)")
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

    config = SmallConfig() if args.config == "small" else MediumConfig()
    model  = ChessTransformer(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt['step']}")

    tokenizer = ChessTokenizer()

    games = load_test_games(args.test_dir)
    if args.max_games:
        games = games[:args.max_games]

    print(f"Evaluating {len(games):,} games...")
    results = evaluate(model, tokenizer, games, device, config)
    print_results(results)


if __name__ == "__main__":
    main()
