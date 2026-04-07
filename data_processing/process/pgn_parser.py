"""
pgn_parser.py — Parse Lichess .pgn.zst dumps into JSONL format.

Streams and decompresses a Lichess database file, filters to a specific
time control category (blitz or rapid), and writes one JSON record per game.

Usage (on EC2):
    python pgn_parser.py \\
        --dump_file ~/data/lichess-raw/2026-01.pgn.zst \\
        --output_file ~/data/lichess-2026-rapid/2026-01.jsonl \\
        --mode rapid \\
        --n_procs 8

Output format (one JSON per line):
    {
        "game-id":       "https://lichess.org/abcd1234",
        "moves-uci":     "e2e4 e7e5 g1f3 ...",
        "moves-seconds": [3, 12, 8, ...],
        "event":         "Rated Rapid game",
        "result":        "1-0",
        "white-elo":     "1654",
        "black-elo":     "1701",
        "termination":   "Normal",
        "time-control":  "600+0",
        "opening":       "Sicilian Defense"
    }

This format is identical to ALLIE's blitz data, so the same tokenizer
can handle both blitz and rapid games.
"""

import argparse
import io
import json
import os
import time
from multiprocessing import Process, Queue, Value
from os.path import dirname

import zstandard as zstd
from chess import pgn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Time control sets
# These were determined by scanning the actual 2026-01.pgn.zst distribution.
# Only games with these time controls are kept; others are discarded.
# ---------------------------------------------------------------------------

RAPID_TIME_CONTROLS = {
    "600+0",   # 10,861,492 games — dominant (77% of rapid)
    "600+5",   # 2,355,396
    "900+10",  # 645,557
    "900+0",   # 214,783
    "300+5",   # 125,149  (classified as rapid by Lichess due to increment)
    "480+0",   # 121,062
    "600+3",   # 109,499
    "600+2",   # 97,877
    "420+2",   # 66,273
    "1200+0",  # 52,665
}

BLITZ_TIME_CONTROLS = {
    # These match ALLIE's token list exactly — same 24 time controls
    "180+0", "300+0", "180+2", "300+3", "300+2", "420+0",
    "240+0", "180+1", "180+3", "300+1", "360+0", "300+4",
    "420+1", "240+2", "180+5", "120+2", "120+3", "240+3",
    "240+1", "360+2", "60+3",  "60+5",  "240+4", "240+5",
}


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------

def parse_time_control(tc: str) -> tuple[int, int]:
    """
    Parse a Lichess time control string into (base_seconds, increment_seconds).

    Example: "600+5" -> (600, 5)
    """
    try:
        base, increment = map(int, tc.split("+"))
        return base, increment
    except Exception:
        raise ValueError(f"Cannot parse time control: {tc!r}")


def extract_moves_and_times(game) -> tuple[str, list[int]]:
    """
    Walk through a parsed game tree and extract moves + time spent per move.

    How clock times work in Lichess PGN:
        Each move has a comment like { [%clk 0:09:45] } showing time REMAINING
        after that move was played. We compute time SPENT as:
            time_spent = clock_before - clock_after + increment

    Returns:
        moves_uci:   space-separated UCI move string, e.g. "e2e4 e7e5 g1f3"
        move_times:  list of ints — seconds spent on each move (same length)
    """
    base_time, increment = parse_time_control(game.headers["TimeControl"])

    # Both players start with full base time on their clock
    clock = [base_time, base_time]  # index 0 = white, 1 = black
    turn = 0  # white moves first

    moves = []
    move_times = []

    node = game.next()
    while node:
        # node.clock() gives seconds remaining after this move was played
        remaining = int(node.clock())
        time_spent = clock[turn] - remaining + increment
        clock[turn] = remaining

        moves.append(node.move.uci())
        move_times.append(max(0, time_spent))  # clamp to 0 (can go negative on lag)

        node = node.next()
        turn = 1 - turn  # alternate between white (0) and black (1)

    return " ".join(moves), move_times


def game_to_dict(game, valid_time_controls: set) -> dict | None:
    """
    Convert a python-chess Game object into our JSONL dict format.

    Returns None if the game should be filtered out (wrong time control,
    missing Elo, too short, or any parse error).
    """
    headers = game.headers
    time_control = headers.get("TimeControl", "")

    # --- Filter 1: must be one of our target time controls ---
    if time_control not in valid_time_controls:
        return None

    # --- Filter 2: both players must have a valid numeric Elo ---
    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except (KeyError, ValueError):
        return None

    # --- Filter 3: game must have clock annotations on every move ---
    # Without clocks we can't extract move times, which we need for training
    try:
        moves_uci, move_times = extract_moves_and_times(game)
    except Exception:
        return None

    # --- Filter 4: skip very short games (likely aborted/disconnected) ---
    if len(moves_uci.split()) <= 8:
        return None

    return {
        "game-id":       headers.get("Site", ""),
        "moves-uci":     moves_uci,
        "moves-seconds": move_times,
        "event":         headers.get("Event", ""),
        "result":        headers.get("Result", ""),
        "white-elo":     str(white_elo),
        "black-elo":     str(black_elo),
        "termination":   headers.get("Termination", ""),
        "time-control":  time_control,
        "opening":       headers.get("Opening", ""),
    }


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------
# We use 3 process types:
#   1. Main thread:     streams + decompresses the .pgn.zst file, feeds raw
#                       PGN strings into in_queue
#   2. Parser workers:  pull raw PGN strings from in_queue, parse + filter,
#                       push JSONL strings to out_queue
#   3. Writer worker:   pulls JSONL strings from out_queue, writes to file
#
# This pipeline keeps all CPUs busy — decompression, parsing, and writing
# happen in parallel rather than sequentially.
# ---------------------------------------------------------------------------

def parse_worker(
    in_queue: Queue,
    out_queue: Queue,
    shutdown: Value,
    valid_time_controls: set,
):
    """Parser worker: PGN string -> JSONL string."""
    while True:
        try:
            game_string = in_queue.get_nowait()
            try:
                game = pgn.read_game(io.StringIO(game_string))
                if game is None:
                    continue
                result = game_to_dict(game, valid_time_controls)
                if result is not None:
                    out_queue.put(json.dumps(result) + "\n")
            except Exception:
                pass  # silently skip malformed games
        except Exception:
            if in_queue.empty() and shutdown.value:
                return
            time.sleep(0.0001)


def write_worker(output_file: str, out_queue: Queue, shutdown: Value):
    """Writer worker: JSONL string -> disk."""
    os.makedirs(dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        while True:
            try:
                line = out_queue.get_nowait()
                f.write(line)
            except Exception:
                if out_queue.empty() and shutdown.value:
                    return
                time.sleep(0.001)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse a Lichess .pgn.zst dump to JSONL"
    )
    parser.add_argument(
        "--dump_file", required=True,
        help="Path to the input .pgn.zst file",
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Path to the output .jsonl file",
    )
    parser.add_argument(
        "--mode", choices=["blitz", "rapid"], default="rapid",
        help="Which time controls to keep (default: rapid)",
    )
    parser.add_argument(
        "--n_procs", type=int, default=8,
        help="Number of parallel parser workers (default: 8)",
    )
    args = parser.parse_args()

    valid_tcs = RAPID_TIME_CONTROLS if args.mode == "rapid" else BLITZ_TIME_CONTROLS
    print(f"Mode: {args.mode} | Keeping {len(valid_tcs)} time controls | {args.n_procs} workers")

    in_queue  = Queue(maxsize=1024)
    out_queue = Queue(maxsize=1024)
    parser_shutdown = Value("d", 0)
    writer_shutdown = Value("d", 0)

    # Start parser workers
    workers = []
    for _ in range(args.n_procs):
        p = Process(
            target=parse_worker,
            args=(in_queue, out_queue, parser_shutdown, valid_tcs),
        )
        p.start()
        workers.append(p)

    # Start writer
    writer = Process(
        target=write_worker,
        args=(args.output_file, out_queue, writer_shutdown),
    )
    writer.start()

    # Main thread: stream-decompress and feed games into the queue
    dctx = zstd.ZstdDecompressor()
    with tqdm(desc="Streaming games") as pbar:
        with open(args.dump_file, "rb") as fb:
            reader     = dctx.stream_reader(fb)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            lines = []
            for line in text_stream:
                # A new [Event ...] line signals the start of a new game.
                # When we see one, ship the accumulated lines as one game string.
                if line.startswith("[Event") and lines:
                    in_queue.put("".join(lines))
                    lines = []
                    pbar.update(1)
                lines.append(line)

            # Flush the final game
            if lines:
                in_queue.put("".join(lines))
                pbar.update(1)

    # Shut down workers gracefully
    parser_shutdown.value = 1
    for p in workers:
        p.join()

    writer_shutdown.value = 1
    writer.join()

    print(f"Done! Output written to: {args.output_file}")


if __name__ == "__main__":
    main()
