"""Vectorized board representation and legal move masking."""

import numpy as np
import chess
from .constants import NUM_PLANES, BOARD_SIZE, ACTION_SPACE_SIZE, PIECE_TYPES, move_to_action

def encode_board(board: chess.Board) -> np.ndarray:
    """Encodes a chess.Board state into a (19, 8, 8) float32 tensor.
    
    Planes layout:
      0..5   : White pieces (P, N, B, R, Q, K)
      6..11  : Black pieces (p, n, b, r, q, k)
      12     : Turn (1.0 for White, 0.0 for Black)
      13     : White Kingside Castling Right
      14     : White Queenside Castling Right
      15     : Black Kingside Castling Right
      16     : Black Queenside Castling Right
      17     : En Passant target square
      18     : Halfmove clock normalized (x / 100.0)
    """
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # 1. Piece positions
    for p_idx, piece_type in enumerate(PIECE_TYPES):
        # White
        white_squares = board.pieces(piece_type, chess.WHITE)
        for sq in white_squares:
            rank, file = divmod(sq, 8)
            tensor[p_idx, rank, file] = 1.0

        # Black
        black_squares = board.pieces(piece_type, chess.BLACK)
        for sq in black_squares:
            rank, file = divmod(sq, 8)
            tensor[p_idx + 6, rank, file] = 1.0

    # 2. Turn
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # 3. Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # 4. En passant
    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        tensor[17, ep_rank, ep_file] = 1.0

    # 5. Halfmove clock (50-move rule counter)
    tensor[18, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return tensor

def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean array of length 4096 indicating legal moves."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        action = move_to_action(move)
        mask[action] = True
    return mask
