"""Direct Lichess Bot API Client for NovaZero.

Connects directly to the Lichess HTTP/NDJSON streaming API to play live games
against real human players and other bots on lichess.org.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import threading
import torch
import chess
import numpy as np

from .constants import action_to_move
from .model import NovaZeroNet
from .mcts import BQSMCTS

LICHESS_HOST = "https://lichess.org"

class LichessBot:
    """Direct Lichess Bot API Interface."""
    def __init__(self, token: str, checkpoint_path: str = None, simulations: int = 40, device_str: str = "auto"):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "NovaZero-Chess-RL/1.0",
        }
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = NovaZeroNet().to(self.device)
        self.simulations = simulations

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[Lichess] Loading neural weights from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("[Lichess] Weights loaded successfully.")
        else:
            print("[Lichess] Warning: No checkpoint found, using randomly initialized weights.")

        self.mcts = BQSMCTS(self.model, self.device, num_simulations=self.simulations)

    def _api_request(self, endpoint: str, data: dict = None, method: str = "GET"):
        url = f"{LICHESS_HOST}{endpoint}"
        req_data = json.dumps(data).encode("utf-8") if data else None
        headers = dict(self.headers)
        if data:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        return urllib.request.urlopen(req)

    def verify_account(self) -> str:
        """Verifies that the bot token is active."""
        try:
            with self._api_request("/api/account") as resp:
                data = json.loads(resp.read().decode("utf-8"))
                username = data.get("username", "Unknown")
                is_bot = data.get("title") == "BOT"
                print(f"[Lichess] Connected as: {username} (BOT status: {is_bot})")
                if not is_bot:
                    print("[Lichess] Note: Account is not marked as BOT. Upgrade at https://lichess.org/api#tag/Bot/operation/botAccountUpgrade")
                return username
        except Exception as e:
            print(f"[Lichess] Authentication failed: {e}")
            sys.exit(1)

    def start_event_stream(self):
        """Streams incoming challenges and game invitations."""
        self.verify_account()
        print("[Lichess] Listening for incoming challenges and games...")

        url = f"{LICHESS_HOST}/api/stream/event"
        req = urllib.request.Request(url, headers=self.headers)

        while True:
            try:
                with urllib.request.urlopen(req) as resp:
                    for line in resp:
                        if not line.strip():
                            continue
                        event = json.loads(line.decode("utf-8"))
                        event_type = event.get("type")

                        if event_type == "challenge":
                            self._handle_challenge(event["challenge"])
                        elif event_type == "gameStart":
                            game_id = event["game"]["gameId"]
                            # Spawn game handler in separate thread
                            t = threading.Thread(target=self._play_game, args=(game_id,), daemon=True)
                            t.start()
            except Exception as e:
                print(f"[Lichess] Stream error: {e}. Reconnecting in 5s...")
                time.sleep(5)

    def _handle_challenge(self, challenge: dict):
        challenge_id = challenge["id"]
        challenger = challenge.get("challenger", {}).get("name", "Unknown")
        variant = challenge.get("variant", {}).get("key", "standard")

        if variant != "standard":
            print(f"[Lichess] Declining non-standard challenge from {challenger} ({variant})")
            try:
                self._api_request(f"/api/challenge/{challenge_id}/decline", method="POST")
            except Exception:
                pass
            return

        print(f"[Lichess] Accepting challenge from {challenger} (ID: {challenge_id})")
        try:
            self._api_request(f"/api/challenge/{challenge_id}/accept", method="POST")
        except Exception as e:
            print(f"[Lichess] Failed to accept challenge: {e}")

    def _play_game(self, game_id: str):
        """Streams game events and plays moves."""
        print(f"\n[Game {game_id}] Started!")
        url = f"{LICHESS_HOST}/api/bot/game/stream/{game_id}"
        req = urllib.request.Request(url, headers=self.headers)

        my_color = None
        board = chess.Board()

        try:
            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    msg = json.loads(line.decode("utf-8"))
                    msg_type = msg.get("type", "gameFull")

                    if msg_type == "gameFull":
                        white_user = msg["white"].get("name", "white")
                        black_user = msg["black"].get("name", "black")
                        # Determine bot color
                        account_name = self.verify_account()
                        my_color = chess.WHITE if white_user.lower() == account_name.lower() else chess.BLACK
                        color_str = "White" if my_color == chess.WHITE else "Black"
                        print(f"[Game {game_id}] Playing as {color_str} vs {black_user if my_color == chess.WHITE else white_user}")
                        
                        # Process existing moves
                        state = msg["state"]
                        moves_str = state.get("moves", "").strip()
                        board = chess.Board()
                        if moves_str:
                            for m in moves_str.split():
                                board.push_uci(m)

                        if board.turn == my_color and not board.is_game_over():
                            self._make_move(game_id, board)

                    elif msg_type == "gameState":
                        moves_str = msg.get("moves", "").strip()
                        board = chess.Board()
                        if moves_str:
                            for m in moves_str.split():
                                board.push_uci(m)

                        status = msg.get("status")
                        if status != "started":
                            print(f"[Game {game_id}] Game finished with status: {status}")
                            break

                        if board.turn == my_color and not board.is_game_over():
                            self._make_move(game_id, board)

                    elif msg_type == "chatLine":
                        pass

        except Exception as e:
            print(f"[Game {game_id}] Ended or connection error: {e}")

    def _make_move(self, game_id: str, board: chess.Board):
        """Computes best move via MCTS and submits to Lichess."""
        if board.is_game_over() or len(list(board.legal_moves)) == 0:
            return

        pi, q = self.mcts.search(board, add_noise=False)
        action = int(np.argmax(pi))
        move = action_to_move(action, board)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))

        uci_move = move.uci()
        print(f"[Game {game_id}] Playing move: {uci_move} (Eval Q: {q:+.2f})")

        try:
            self._api_request(f"/api/bot/game/{game_id}/move/{uci_move}", method="POST")
        except Exception as e:
            print(f"[Game {game_id}] Failed to send move {uci_move}: {e}")
