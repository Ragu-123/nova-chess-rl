"""Constants and action space mapping for NovaZero."""

import chess

NUM_PLANES = 19
BOARD_SIZE = 8
ACTION_SPACE_SIZE = 4096  # from_sq (0..63) * 64 + to_sq (0..63)

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

def move_to_action(move: chess.Move) -> int:
    """Encodes a chess.Move into a flat action index in [0, 4095]."""
    return move.from_square * 64 + move.to_square

def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Decodes a flat action index into a legal chess.Move.
    
    Handles pawn promotions to Queen by default, or Knight if queen is illegal.
    """
    from_sq = action // 64
    to_sq = action % 64
    
    # Try standard move first
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move
        
    # Check for promotion moves
    for promo_piece in [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]:
        promo_move = chess.Move(from_sq, to_sq, promotion=promo_piece)
        if promo_move in board.legal_moves:
            return promo_move
            
    # Fallback to standard move (will be caught by legality validator)
    return move
