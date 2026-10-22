"""Evaluation and benchmarking module for NovaZero."""

import os
import random
import numpy as np
import torch
import chess
from .constants import action_to_move
from .mcts import BQSMCTS
from .video_recorder import ChessVideoRecorder

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

def evaluate_material(board: chess.Board) -> int:
    """Computes basic material balance from White's perspective."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
    return score

class GreedyMaterialPlayer:
    """Fast baseline: 1-ply greedy material evaluation."""
    def choose_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = evaluate_material(board)
            board.pop()
            
            if board.turn == chess.WHITE:
                if score > best_eval:
                    best_eval = score
                    best_move = move
            else:
                if score < best_eval:
                    best_eval = score
                    best_move = move
                    
        return best_move if best_move else random.choice(list(board.legal_moves))


def run_tournament(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = 10,
    simulations: int = 40,
    record_video: bool = True,
    output_dir: str = "outputs",
) -> dict:
    """Evaluates NovaZero against a Greedy Material baseline over N games."""
    mcts = BQSMCTS(model, device, num_simulations=simulations)
    greedy = GreedyMaterialPlayer()
    recorder = ChessVideoRecorder(output_dir=os.path.join(output_dir, "eval_videos"), fps=2)

    wins = 0
    draws = 0
    losses = 0

    print(f"\n=== Starting Tournament: NovaZero vs. Greedy Material ({num_games} games) ===")

    for game_idx in range(1, num_games + 1):
        board = chess.Board()
        novazero_color = chess.WHITE if (game_idx % 2 == 1) else chess.BLACK
        
        rec = recorder.maybe_start_recording(game_idx, probability=1.0 if (record_video and game_idx == 1) else 0.0)
        step = 0
        last_move = None
        eval_score = 0.0

        while not board.is_game_over() and step < 90:
            if board.turn == novazero_color:
                pi, eval_score = mcts.search(board, add_noise=False)
                action = int(np.argmax(pi))
                move = action_to_move(action, board)
                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
            else:
                move = greedy.choose_move(board)

            if rec:
                recorder.capture_frame(board, last_move=last_move, step=step, eval_score=eval_score)

            board.push(move)
            last_move = move
            step += 1

        # Score
        if board.is_checkmate():
            if board.turn != novazero_color:
                wins += 1
                result_str = "WIN (Checkmate)"
            else:
                losses += 1
                result_str = "LOSS"
        else:
            # Draw / Step limit
            draws += 1
            result_str = "DRAW"

        video_path = ""
        if rec:
            recorder.capture_frame(board, last_move=last_move, step=step, eval_score=eval_score)
            video_path = recorder.save_mp4(f"eval_game_{game_idx:02d}.mp4")

        color_str = "White" if novazero_color == chess.WHITE else "Black"
        print(f"Game {game_idx:02d}: NovaZero ({color_str}) -> {result_str} in {step} plies {f'[Saved: {video_path}]' if video_path else ''}")

    total_score = wins + 0.5 * draws
    win_rate = (total_score / num_games) * 100.0
    print(f"\nFinal Score: {wins}W - {losses}L - {draws}D ({win_rate:.1f}%)")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}
