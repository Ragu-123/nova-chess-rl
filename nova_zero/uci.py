"""Universal Chess Interface (UCI) Protocol Engine for NovaZero.

Fully compliant with:
- lichess-bot (https://github.com/lichess-bot-devs/lichess-bot)
- cutechess-cli & fastchess tournament runners
- Arena, ChessBase, Banksia GUI
"""

import sys
import os
import torch
import chess
import numpy as np

from .constants import action_to_move
from .model import NovaZeroNet
from .mcts import BQSMCTS

class UCIEngine:
    """UCI Protocol Handler for NovaZero."""
    def __init__(self, checkpoint_path: str = None, device_str: str = "auto", simulations: int = 50):
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = NovaZeroNet().to(self.device)
        self.simulations = simulations
        self.board = chess.Board()

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        self.mcts = BQSMCTS(self.model, self.device, num_simulations=self.simulations)

    def run(self):
        """Main UCI command loop reading from stdin and writing to stdout."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self._handle_command(line)
            except (KeyboardInterrupt, SystemExit):
                break

    def _handle_command(self, cmd_line: str):
        tokens = cmd_line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name NovaZero-Chess")
            print("id author NovaZero-RL")
            print("option name Simulations type spin default 50 min 1 max 2000")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "setoption":
            # e.g., setoption name Simulations value 100
            if len(tokens) >= 5 and tokens[2].lower() == "simulations":
                try:
                    self.simulations = int(tokens[4])
                    self.mcts.num_simulations = self.simulations
                except ValueError:
                    pass

        elif cmd == "ucinewgame":
            self.board = chess.Board()

        elif cmd == "position":
            # Syntax: position [fen <FEN> | startpos] moves <move1> ...
            self._parse_position(tokens[1:])

        elif cmd == "go":
            self._handle_go(tokens[1:])

        elif cmd == "quit":
            sys.exit(0)

    def _parse_position(self, tokens: list):
        if not tokens:
            return

        idx = 0
        if tokens[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif tokens[0] == "fen":
            fen_parts = []
            idx = 1
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))

        if idx < len(tokens) and tokens[idx] == "moves":
            for move_str in tokens[idx + 1:]:
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)

    def _handle_go(self, tokens: list):
        """Runs search and outputs bestmove."""
        if self.board.is_game_over() or len(list(self.board.legal_moves)) == 0:
            print("bestmove 0000")
            sys.stdout.flush()
            return

        pi, root_q = self.mcts.search(self.board, add_noise=False)
        action = int(np.argmax(pi))
        best_move = action_to_move(action, self.board)

        if best_move not in self.board.legal_moves:
            best_move = next(iter(self.board.legal_moves))

        print(f"bestmove {best_move.uci()}")
        sys.stdout.flush()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NovaZero UCI Engine")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    engine = UCIEngine(
        checkpoint_path=args.checkpoint,
        device_str=args.device,
        simulations=args.simulations
    )
    engine.run()

if __name__ == "__main__":
    main()
