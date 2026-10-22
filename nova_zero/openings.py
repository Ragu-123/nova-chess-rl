"""Grandmaster opening book and master move seedings for NovaZero.

Provides curated opening lines to seed self-play training, preventing
the agent from wasting thousands of episodes rediscovering standard opening principles.
"""

import random
import chess

MASTER_OPENING_LINES = [
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"],
    # Sicilian Defense (Open / Najdorf)
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Sicilian Defense (Classical)
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "d7d6"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"],
    # Caro-Kann Defense
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],
    # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
    # Queen's Gambit Accepted
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    # Nimzo-Indian Defense
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"],
    # English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "d7d5"],
    # Reti Opening
    ["g1f3", "d7d5", "c2c4", "c7c6", "b2b3", "g8f6", "c1b2", "c8f5"],
]

def get_opening_move(board: chess.Board, step: int, seed_prob: float = 0.40) -> chess.Move:
    """Returns a master opening move if available and within opening horizon (first 6 plies)."""
    if step >= 8 or random.random() > seed_prob:
        return None

    # Find matching opening lines with current move stack
    played_uci = [m.uci() for m in board.move_stack]
    candidates = []

    for line in MASTER_OPENING_LINES:
        if len(line) > step and line[:step] == played_uci:
            next_move_uci = line[step]
            candidate_move = chess.Move.from_uci(next_move_uci)
            if candidate_move in board.legal_moves:
                candidates.append(candidate_move)

    if candidates:
        return random.choice(candidates)

    return None
