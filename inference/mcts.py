"""
mcts.py — Monte Carlo Tree Search for human-aligned chess move selection.

Uses the trained ChessTransformer's three heads:
  - policy_head  → prior probabilities over legal moves (guides exploration)
  - value_head   → position evaluation in [-1, 1] (replaces random rollouts)
  - time_head    → predicted move time (used for human-likeness weighting)

HOW IT WORKS
------------
Standard AlphaZero-style MCTS with Elo conditioning:

1. SELECT   — from the root, repeatedly pick the child with the highest UCB score:
              UCB(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
              Q = mean value from past simulations
              P = policy prior from the neural network
              N = visit counts

2. EXPAND   — when we reach a leaf node (unvisited position), call the model to:
              a) get policy priors for all legal moves
              b) get a value estimate for this position

3. BACKPROP — propagate the value estimate back up through all visited nodes,
              updating visit counts and mean values

4. MOVE     — after all simulations, pick the move with the most visits
              (most robust estimator, less sensitive to outliers than highest Q)

ELO CONDITIONING
----------------
The entire search is conditioned on the target Elo. The model sees the current
game sequence (with Elo tokens prepended) and predicts moves that a player of
that Elo level would make. This is what makes it "human-aligned" rather than
just a strong engine.

TIME WEIGHTING (optional)
--------------------------
If use_time_weighting=True, we multiply the policy prior by a human-likeness
factor derived from the time head — moves the model predicts a human would
spend more time on are slightly boosted. This follows ALLIE's FULL variant.

Usage:
    import chess
    from inference.mcts import MCTS

    mcts = MCTS(model, tokenizer, num_simulations=200, c_puct=1.5)
    board = chess.Board()
    move = mcts.get_move(
        board=board,
        move_history=[],        # list of UCI strings played so far
        white_elo=1500,
        black_elo=1500,
        time_control="600+0",
    )
    print(move)  # e.g. "e2e4"
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from data_processing.files.chess_moves import CHESS_MOVES
from tokenizer.chess_tokenizer import ChessTokenizer

# Build a lookup: UCI string → token ID (index in CHESS_MOVES)
MOVE_TO_ID = {move: i for i, move in enumerate(CHESS_MOVES)}
ID_TO_MOVE = {i: move for i, move in enumerate(CHESS_MOVES)}
N_MOVES = len(CHESS_MOVES)  # 1968


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    One node in the MCTS tree — represents a chess position.

    Each node corresponds to a specific board state reached by a sequence of
    moves from the root. The node stores statistics accumulated across all
    simulations that passed through it.
    """

    __slots__ = (
        "move",         # UCI string that led to this node (None for root)
        "parent",       # parent MCTSNode (None for root)
        "prior",        # P(s, a) — policy prior from the neural network
        "visit_count",  # N(s, a) — how many simulations passed through here
        "value_sum",    # sum of value estimates from all simulations
        "children",     # dict: UCI string → MCTSNode
        "is_expanded",  # True after we've called the model on this node
    )

    def __init__(self, move: Optional[str], parent: Optional["MCTSNode"], prior: float):
        self.move        = move
        self.parent      = parent
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self.children: Dict[str, "MCTSNode"] = {}
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean value estimate. Returns 0 for unvisited nodes."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        Upper Confidence Bound score for node selection.

        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        The first term exploits known good moves.
        The second term explores less-visited moves weighted by their prior.
        """
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
        )
        return self.q_value + exploration

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def __repr__(self) -> str:
        return (
            f"MCTSNode(move={self.move}, N={self.visit_count}, "
            f"Q={self.q_value:.3f}, prior={self.prior:.3f})"
        )


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """
    Monte Carlo Tree Search using a trained ChessTransformer.

    The search is fully Elo-conditioned — the model predicts moves consistent
    with a player of the specified rating, not necessarily the strongest move.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ChessTokenizer,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        use_time_weighting: bool = True,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model:               trained ChessTransformer (in eval mode)
            tokenizer:           ChessTokenizer instance
            num_simulations:     number of MCTS simulations per move (more = stronger)
            c_puct:              exploration constant (higher = more exploration)
            use_time_weighting:  if True, boost priors for moves with higher predicted time
                                 (follows ALLIE-FULL; set False for pure policy MCTS)
            temperature:         temperature for final move selection (1.0=proportional
                                 to visits, 0.0=argmax, used for training self-play)
            device:              torch device (auto-detected if None)
        """
        self.model            = model
        self.tokenizer        = tokenizer
        self.num_simulations  = num_simulations
        self.c_puct           = c_puct
        self.use_time_weighting = use_time_weighting
        self.temperature      = temperature
        self.device           = device or next(model.parameters()).device

        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_move(
        self,
        board: chess.Board,
        move_history: List[str],
        white_elo: int,
        black_elo: int,
        time_control: str = "600+0",
        max_seq_len: int = 512,
    ) -> str:
        """
        Run MCTS and return the best move UCI string for the current position.

        Args:
            board:        current python-chess Board object
            move_history: list of UCI strings played so far (e.g. ["e2e4", "e7e5"])
            white_elo:    white player's Elo rating
            black_elo:    black player's Elo rating
            time_control: time control string (e.g. "600+0")
            max_seq_len:  maximum context length for the transformer

        Returns:
            UCI string of the selected move (e.g. "e2e4")
        """
        if board.is_game_over():
            raise ValueError("Game is already over — no move to make.")

        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Root node gets uniform prior — will be overwritten on first expansion
        root = MCTSNode(move=None, parent=None, prior=1.0)

        # Build the token context for the current position
        context_ids = self._build_context(
            move_history, white_elo, black_elo, time_control, max_seq_len
        )

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, board.copy(), move_history.copy(), context_ids, max_seq_len)

        # Select best move
        return self._select_move(root, legal_moves)

    # ------------------------------------------------------------------
    # Core MCTS logic
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: MCTSNode,
        board: chess.Board,
        move_history: List[str],
        context_ids: List[int],
        max_seq_len: int,
    ):
        """Run one full simulation: select → expand → evaluate → backpropagate."""
        node = root
        path: List[MCTSNode] = [node]
        sim_board = board
        sim_history = move_history.copy()

        # --- SELECT: traverse tree until we find a leaf or terminal ---
        while not node.is_leaf() and not sim_board.is_game_over():
            legal_uci = {m.uci() for m in sim_board.legal_moves}

            # Filter children to legal moves only
            legal_children = {
                m: c for m, c in node.children.items() if m in legal_uci
            }
            if not legal_children:
                break

            # Pick child with highest UCB score
            best_move = max(
                legal_children,
                key=lambda m: legal_children[m].ucb_score(node.visit_count, self.c_puct),
            )
            node = legal_children[best_move]
            path.append(node)

            # Apply move to board
            sim_board.push_uci(best_move)
            sim_history.append(best_move)

        # --- EVALUATE: get value for this position ---
        if sim_board.is_game_over():
            value = self._terminal_value(sim_board)
        else:
            # EXPAND: call the model to get policy priors and value
            legal_uci = [m.uci() for m in sim_board.legal_moves]
            priors, value = self._evaluate_position(
                sim_history, legal_uci, context_ids, max_seq_len
            )

            # Add children for all legal moves
            for move, prior in priors.items():
                node.children[move] = MCTSNode(
                    move=move, parent=node, prior=prior
                )
            node.is_expanded = True

        # --- BACKPROPAGATE: update all nodes on the path ---
        # Value is from the perspective of the player to move at that node.
        # We flip the sign at each level since players alternate.
        for i, n in enumerate(reversed(path)):
            # Flip value for alternating players
            signed_value = value if i % 2 == 0 else -value
            n.visit_count += 1
            n.value_sum   += signed_value

    def _evaluate_position(
        self,
        move_history: List[str],
        legal_moves: List[str],
        base_context: List[int],
        max_seq_len: int,
    ) -> Tuple[Dict[str, float], float]:
        """
        Call the model on the current position to get:
          - A prior probability for each legal move
          - A scalar value estimate in [-1, 1]

        Returns:
            priors: dict mapping UCI move string → probability
            value:  float in [-1, 1]
        """
        # Build token sequence for current history
        context = self._build_context_from_ids(
            base_context, move_history, max_seq_len
        )

        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Policy logits at the last position → probabilities over all moves
        last_logits = outputs["policy_logits"][0, -1, :N_MOVES]  # (1968,)
        last_value  = float(outputs["value_pred"][0, -1])
        last_time   = outputs["time_pred"][0, -1]  # (scalar)

        # Mask out illegal moves — set their logits to -inf
        legal_ids = set()
        for uci in legal_moves:
            if uci in MOVE_TO_ID:
                legal_ids.add(MOVE_TO_ID[uci])

        mask = torch.full((N_MOVES,), float("-inf"), device=self.device)
        for idx in legal_ids:
            mask[idx] = 0.0
        masked_logits = last_logits + mask

        # Softmax over legal moves
        probs = F.softmax(masked_logits, dim=-1)  # (1968,)

        # Optional: boost priors by predicted time (ALLIE-FULL style)
        if self.use_time_weighting:
            # time_pred is a single value for the position; we use it as a
            # scalar confidence boost for moves the model is more "thoughtful" about
            # Here we apply a mild boost proportional to predicted think time
            time_weight = float(torch.sigmoid(last_time))
            # Blend: 80% policy prior, 20% time-weighted prior
            probs = 0.8 * probs + 0.2 * (probs * time_weight)
            probs = probs / probs.sum()

        # Build priors dict for legal moves only
        probs_np = probs.cpu().numpy()
        priors: Dict[str, float] = {}
        for uci in legal_moves:
            if uci in MOVE_TO_ID:
                priors[uci] = float(probs_np[MOVE_TO_ID[uci]])
            else:
                priors[uci] = 1.0 / len(legal_moves)  # fallback for rare moves

        # Normalize (in case of floating point drift)
        total = sum(priors.values())
        if total > 0:
            priors = {m: p / total for m, p in priors.items()}

        return priors, last_value

    def _terminal_value(self, board: chess.Board) -> float:
        """Return the game value for a terminal position."""
        result = board.result()
        if result == "1-0":
            return 1.0   # white wins
        elif result == "0-1":
            return -1.0  # black wins
        else:
            return 0.0   # draw

    def _select_move(self, root: MCTSNode, legal_moves: List[str]) -> str:
        """
        After all simulations, select the final move.

        With temperature=0: pick the most-visited move (strongest play).
        With temperature=1: sample proportionally to visit counts (more human-like).
        """
        if not root.children:
            # Fallback: pick a random legal move if tree is empty
            return np.random.choice(legal_moves)

        legal_children = {
            m: c for m, c in root.children.items() if m in legal_moves
        }
        if not legal_children:
            return np.random.choice(legal_moves)

        moves   = list(legal_children.keys())
        visits  = np.array([legal_children[m].visit_count for m in moves], dtype=np.float64)

        if self.temperature == 0.0 or visits.sum() == 0:
            # Argmax
            return moves[int(visits.argmax())]
        else:
            # Sample proportional to visit_count^(1/temperature)
            visits = visits ** (1.0 / self.temperature)
            visits /= visits.sum()
            return np.random.choice(moves, p=visits)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        move_history: List[str],
        white_elo: int,
        black_elo: int,
        time_control: str,
        max_seq_len: int,
    ) -> List[int]:
        """
        Build the token ID sequence for the game so far.
        Format: [white_elo_token, black_elo_token, tc_token, move1, move2, ...]
        """
        vocab_size = self.tokenizer.vocab_size

        # Elo tokens are stored as elo + vocab_size
        white_tok = white_elo + vocab_size
        black_tok = black_elo + vocab_size

        # Time control token
        tc_id = self.tokenizer.time_control_to_id(time_control)

        ids = [white_tok, black_tok, tc_id]

        # Add moves
        for uci in move_history:
            if uci in MOVE_TO_ID:
                ids.append(MOVE_TO_ID[uci])

        # Truncate to max_seq_len from the right (keep most recent context)
        return ids[-max_seq_len:]

    def _build_context_from_ids(
        self,
        base_context: List[int],
        move_history: List[str],
        max_seq_len: int,
    ) -> List[int]:
        """
        Extend the base context with additional moves from the current simulation.
        The base context already contains Elo + TC + moves up to the root position.
        We append any additional moves made during simulation.
        """
        # base_context already has moves up to root; move_history may be longer
        # Figure out how many moves were in the original position
        # (base_context includes elo tokens + tc token + original moves)
        extra_moves = move_history[len(base_context) - 3:]  # subtract 3 header tokens

        ids = base_context.copy()
        for uci in extra_moves:
            if uci in MOVE_TO_ID:
                ids.append(MOVE_TO_ID[uci])

        return ids[-max_seq_len:]


# ---------------------------------------------------------------------------
# Convenience: play a full game with MCTS
# ---------------------------------------------------------------------------

def play_game(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    white_elo: int = 1500,
    black_elo: int = 1500,
    time_control: str = "600+0",
    num_simulations: int = 100,
    max_moves: int = 200,
    verbose: bool = True,
) -> str:
    """
    Play a complete game using MCTS for both sides.

    Returns the PGN result string ("1-0", "0-1", or "1/2-1/2").
    """
    mcts = MCTS(model, tokenizer, num_simulations=num_simulations)
    board = chess.Board()
    move_history: List[str] = []

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        t0 = time.time()
        move = mcts.get_move(
            board=board,
            move_history=move_history,
            white_elo=white_elo,
            black_elo=black_elo,
            time_control=time_control,
        )
        elapsed = time.time() - t0

        board.push_uci(move)
        move_history.append(move)

        if verbose:
            side = "White" if move_num % 2 == 0 else "Black"
            print(f"  Move {move_num + 1:3d} ({side}): {move}  [{elapsed:.1f}s, {num_simulations} sims]")

    result = board.result()
    if verbose:
        print(f"\nResult: {result}")
        print(f"Moves:  {' '.join(move_history)}")

    return result
