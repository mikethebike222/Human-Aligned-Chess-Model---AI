"""
chess_tokenizer.py — Converts chess games to/from packed uint32 token arrays.

This tokenizer is compatible with ALLIE's 2022.bin training data format,
meaning games tokenized here can be concatenated directly with that file.

---------------------------------------------------------------------------
HOW THE VOCABULARY WORKS
---------------------------------------------------------------------------
The vocab has named slots:

    [0    – 1967]     chess moves in UCI notation (e.g. "e2e4", "a7a8q")
    [1968 – 1991]     blitz time control tokens   (24 tokens, same as ALLIE)
    [1992 – 2001]     rapid time control tokens   (10 tokens, new)
    [2002]            <RESIGNED-OR-CHECKMATED>
    [2003]            <BOS>
    [2004]            <EOS>
    [2005]            <UNK>
    [2006]            <PAD>

Total named tokens: 2007

ELO is NOT a token in the vocab. It is encoded as:
    elo + vocab_size
stored in bits 13–0. Values >= vocab_size in those bits = ELO values.

---------------------------------------------------------------------------
HOW THE BIT-PACKING WORKS (uint32 per token)
---------------------------------------------------------------------------
Each uint32 encodes THREE pieces of information:

    Bits 31–16  │  Bits 15–14  │  Bits 13–0
    ────────────┼──────────────┼────────────
    move time   │  outcome     │  token ID
    (secs + 1)  │  (all tokens)│  (vocab idx)

  - Move time bits (31-16):
      The number of seconds the player spent on this move, plus 1.
      Value 0 means "this token is not a move" (e.g. BOS, ELO, time ctrl).

  - Outcome bits (15-14):
      The game result, OR'd into EVERY token in the game:
        0x0000 → white wins
        0x4000 → draw
        0x8000 → black wins

  - Token ID bits (13-0):
      The vocabulary index, or ELO + vocab_size for ELO tokens.

Example packed token for "e2e4" (token_id=512) with 5s spent, white winning:
    time_bits    = (5 + 1) << 16 = 0x00060000
    outcome_bits = 0x0000        (white win)
    token_bits   = 512           = 0x00000200
    packed       = 0x00060200
"""

from typing import Dict, List, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Import the moves list — same file as ALLIE's moves.py, already in project.
# Must match ALLIE's order EXACTLY for 2022.bin compatibility.
# ---------------------------------------------------------------------------
from data_processing.files.chess_moves import CHESS_MOVES


# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

# Blitz time controls — same 24 tokens as ALLIE (order must match)
BLITZ_TIME_CONTROLS = [
    "180+0", "300+0", "180+2", "300+3", "300+2", "420+0",
    "240+0", "180+1", "180+3", "300+1", "360+0", "300+4",
    "420+1", "240+2", "180+5", "120+2", "120+3", "240+3",
    "240+1", "360+2", "60+3",  "60+5",  "240+4", "240+5",
]

# Rapid time controls — new tokens added after blitz (order matters for
# patching 2022.bin; most common first for easy debugging)
RAPID_TIME_CONTROLS = [
    "600+0",   # 10,861,492 games
    "600+5",   # 2,355,396
    "900+10",  # 645,557
    "900+0",   # 214,783
    "300+5",   # 125,149
    "480+0",   # 121,062
    "600+3",   # 109,499
    "600+2",   # 97,877
    "420+2",   # 66,273
    "1200+0",  # 52,665
]

TERMINATION_TOKEN = "<RESIGNED-OR-CHECKMATED>"

# Game outcome masks — OR'd into every token in a game
OUTCOME_MASKS = {
    "1-0":     np.uint32(0x0000),  # white wins
    "1/2-1/2": np.uint32(0x4000),  # draw
    "0-1":     np.uint32(0x8000),  # black wins
}
WHITE_WIN_MASK = np.uint32(0x0000)
DRAW_MASK      = np.uint32(0x4000)
BLACK_WIN_MASK = np.uint32(0x8000)

# Time normalization stats — empirical values from ALLIE's blitz training data.
# These will need to be recomputed after combining blitz + rapid data since
# rapid move times are much longer (~20-30s avg vs ~4.6s for blitz).
TIME_MEAN  = 4.64001
TIME_STDEV = 6.16533


# ---------------------------------------------------------------------------
# ChessTokenizer
# ---------------------------------------------------------------------------

class ChessTokenizer:
    """
    Converts between chess game dicts (from pgn_parser.py) and packed
    uint32 numpy arrays for training.

    Key design decisions:
    - Compatible with ALLIE's 2022.bin format (same vocab layout, same packing)
    - Elo is NOT a named token — encoded as elo + vocab_size
    - 5% of games use BOS instead of Elo (Elo dropout for generalization)
    - 5% of games use <UNK> for time control (time control dropout)
    """

    def __init__(self):
        # Build the full token list in order — this order defines token IDs
        # and MUST stay stable once 2022.bin exists (changing order breaks data)
        self._tokens = (
            list(CHESS_MOVES)           # IDs 0–1967
            + BLITZ_TIME_CONTROLS       # IDs 1968–1991
            + RAPID_TIME_CONTROLS       # IDs 1992–2001
            + [TERMINATION_TOKEN]       # ID 2002
            + ["<BOS>", "<EOS>", "<UNK>", "<PAD>"]  # IDs 2003–2006
        )

        # Map token string → integer ID (for fast lookup during tokenization)
        self._token_to_id = {tok: i for i, tok in enumerate(self._tokens)}

        # Named token IDs for convenience
        self.bos_id        = self._token_to_id["<BOS>"]
        self.eos_id        = self._token_to_id["<EOS>"]
        self.unk_id        = self._token_to_id["<UNK>"]
        self.pad_id        = self._token_to_id["<PAD>"]
        self.termination_id = self._token_to_id[TERMINATION_TOKEN]

        # Boundary where ELO encoding starts (IDs >= vocab_size are ELO values)
        # e.g. a player rated 1500 is stored as 1500 + 2007 = 3507 in bits 13-0
        self.vocab_size = len(self._tokens)  # 2007

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def tokenize(
        self,
        game: dict,
        add_elo: bool = True,
        add_time_control: bool = True,
        add_termination: bool = True,
    ) -> np.ndarray:
        """
        Convert a game dict (from pgn_parser or data_process) to a uint32 array.

        Args:
            game:              dict with keys: moves-uci, moves-seconds,
                               white-elo, black-elo, time-control, result,
                               termination
            add_elo:           if False, use <BOS> instead of Elo tokens
                               (ALLIE drops Elo for ~5% of games)
            add_time_control:  if False, use <UNK> instead of time control token
                               (ALLIE drops time ctrl for ~5% of games)
            add_termination:   if True and game ended normally, append the
                               termination token before <EOS>

        Returns:
            1D numpy array of uint32 tokens, ready to append to a .bin file
        """
        moves       = game["moves-uci"].split()
        move_times  = game["moves-seconds"]
        time_ctrl   = game.get("time-control", "")
        result      = game.get("result", "1/2-1/2")
        terminated  = game.get("termination", "") == "Normal"

        # Determine outcome mask — applied to every token in this game
        outcome_mask = OUTCOME_MASKS.get(result, DRAW_MASK)

        # --- Build the token sequence ---
        tokens = []

        # 1. ELO prefix (or BOS if Elo dropout)
        if add_elo:
            white_elo = int(game["white-elo"])
            black_elo = int(game["black-elo"])
            # ELO stored as elo + vocab_size; no move time (upper bits = 0)
            tokens.append(self._pack(white_elo + self.vocab_size, move_time=-1))
            tokens.append(self._pack(black_elo + self.vocab_size, move_time=-1))
        else:
            tokens.append(self._pack(self.bos_id, move_time=-1))

        # 2. Time control token (or UNK if time control dropout)
        if add_time_control:
            tc_id = self._token_to_id.get(time_ctrl, self.unk_id)
        else:
            tc_id = self.unk_id
        # Time control token has no move time — time is stored with each move
        tokens.append(self._pack(tc_id, move_time=-1))

        # 3. Move tokens — each carries its move time in the upper 16 bits
        for move, secs in zip(moves, move_times):
            move_id = self._token_to_id.get(move, self.unk_id)
            tokens.append(self._pack(move_id, move_time=int(secs)))

        # 4. Termination token (if game ended with checkmate or resignation)
        if add_termination and terminated:
            tokens.append(self._pack(self.termination_id, move_time=-1))

        # 5. EOS token
        tokens.append(self._pack(self.eos_id, move_time=-1))

        # Convert to numpy array and apply the outcome mask to every token
        arr = np.array(tokens, dtype=np.uint32)
        arr = arr | outcome_mask
        return arr

    def decode_token(self, packed: np.uint32) -> dict:
        """
        Unpack a single uint32 into its three components.

        Returns a dict with:
            token_id:   vocabulary index (or ELO value if >= vocab_size)
            is_elo:     True if this token encodes an ELO value
            elo:        the ELO value (only valid if is_elo=True)
            token_str:  human-readable token string
            move_time:  seconds spent (-1 if not a move token)
            outcome:    "white" / "draw" / "black"
        """
        token_id  = int(packed & np.uint32(0x3FFF))
        move_time = int(packed >> 16) - 1  # -1 means "not a move"
        outcome_bits = int((packed & np.uint32(0xFFFF)) >> 14)

        outcome_map = {0: "white", 1: "draw", 2: "black"}
        outcome = outcome_map.get(outcome_bits, "unknown")

        is_elo = token_id >= self.vocab_size
        if is_elo:
            elo = token_id - self.vocab_size
            token_str = f"<ELO:{elo}>"
        else:
            elo = None
            token_str = self._tokens[token_id] if token_id < len(self._tokens) else "<INVALID>"

        return {
            "token_id":  token_id,
            "is_elo":    is_elo,
            "elo":       elo,
            "token_str": token_str,
            "move_time": move_time,
            "outcome":   outcome,
        }

    def decode_game(self, tokens: np.ndarray) -> List[dict]:
        """Decode a full token array into a list of human-readable dicts."""
        return [self.decode_token(t) for t in tokens]

    def get_token_id(self, token: str) -> int:
        """Look up a token string's ID. Returns unk_id if not found."""
        return self._token_to_id.get(token, self.unk_id)

    def time_control_to_id(self, time_control: str) -> int:
        """Return the token ID for a time control string (e.g. '600+0')."""
        return self._token_to_id.get(time_control, self.unk_id)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"ChessTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"moves={len(CHESS_MOVES)}, "
            f"blitz_tc={len(BLITZ_TIME_CONTROLS)}, "
            f"rapid_tc={len(RAPID_TIME_CONTROLS)})"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _pack(self, token_id: int, move_time: int) -> np.uint32:
        """
        Pack a token ID and move time into a uint32.

        Args:
            token_id:  vocabulary index (bits 13–0)
            move_time: seconds spent on this move, or -1 if not a move token
                       (e.g. ELO tokens, BOS, time control tokens)

        Returns:
            uint32 with move time in bits 31–16, token ID in bits 13–0.
            Outcome mask is applied separately after all tokens are packed.
        """
        # Upper 16 bits: (seconds + 1), or 0 if this isn't a move token
        time_bits = np.uint32(max(0, move_time + 1)) << np.uint32(16)
        return time_bits | np.uint32(token_id)


# ---------------------------------------------------------------------------
# Time normalization helpers (used by the training DataLoader)
# ---------------------------------------------------------------------------

def normalize_time(seconds: float) -> float:
    """
    Normalize a move time using the training set's mean and stdev.
    Clips to [0, 60] before normalizing to reduce outlier impact.
    After adding rapid data, recompute TIME_MEAN and TIME_STDEV and update above.
    """
    seconds = max(0.0, min(60.0, seconds))
    return (seconds - TIME_MEAN) / TIME_STDEV


def denormalize_time(normalized: float) -> float:
    """Reverse of normalize_time — convert back to seconds."""
    return normalized * TIME_STDEV + TIME_MEAN
