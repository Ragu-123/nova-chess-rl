"""Episodic Video Recorder for Chess Matches.

Renders high-definition chessboard animations with evaluation bars,
move notifications, and exports them directly to MP4 format.
"""

import os
import random
import numpy as np
import cv2
import chess

# Modern Chessboard Palette (Lichess Wood/Tournament style)
COLOR_LIGHT = (210, 238, 238)   # Cream (BGR)
COLOR_DARK = (86, 150, 118)     # Forest Olive (BGR)
COLOR_HIGHLIGHT = (80, 205, 250) # Warm Gold (BGR)
COLOR_BG = (28, 28, 30)         # Dark slate header (BGR)
COLOR_TEXT = (245, 245, 245)
COLOR_WHITE_PIECE = (255, 255, 255)
COLOR_BLACK_PIECE = (30, 30, 30)

UNICODE_PIECES = {
    'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
    'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k'
}

class ChessVideoRecorder:
    """Records full game trajectories and encodes them into MP4 videos."""
    def __init__(self, output_dir: str = "outputs/videos", fps: int = 2, square_size: int = 70, max_videos: int = 10):
        self.output_dir = output_dir
        self.fps = fps
        self.sq_size = square_size
        self.max_videos = max_videos
        self.board_size = square_size * 8
        self.header_height = 80
        self.eval_bar_height = 16
        self.total_width = self.board_size
        self.total_height = self.board_size + self.header_height + self.eval_bar_height
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.frames = []
        self.recording_active = False

    def maybe_start_recording(self, episode_idx: int, probability: float = 0.20, force: bool = False) -> bool:
        """Randomly decides whether to record this episode based on probability."""
        if force or (random.random() < probability):
            self.recording_active = True
            self.frames = []
            return True
        self.recording_active = False
        return False

    def capture_frame(self, board: chess.Board, last_move: chess.Move = None, step: int = 0, eval_score: float = 0.0):
        """Renders the current game position and appends to frame buffer."""
        if not self.recording_active:
            return

        frame = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        frame[:] = COLOR_BG

        # 1. Render Top Header
        turn_str = "White to Move" if board.turn == chess.WHITE else "Black to Move"
        move_str = f"Last Move: {last_move.uci()}" if last_move else "Game Start"
        cv2.putText(frame, f"NovaZero RL Engine | Step {step}", (14, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{turn_str} | {move_str}", (14, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 210, 255), 1, cv2.LINE_AA)

        # 2. Render Evaluation Bar (Value in [-1.0, 1.0])
        # Center = 0, Left = -1 (Black), Right = +1 (White)
        bar_y1 = self.header_height - 12
        bar_y2 = self.header_height - 2
        norm_val = np.clip((eval_score + 1.0) / 2.0, 0.0, 1.0)
        split_x = int(norm_val * self.total_width)
        
        # Black advantage section (left)
        cv2.rectangle(frame, (0, bar_y1), (split_x, bar_y2), (70, 70, 70), -1)
        # White advantage section (right)
        cv2.rectangle(frame, (split_x, bar_y1), (self.total_width, bar_y2), (230, 230, 230), -1)
        # Center mark
        center_x = self.total_width // 2
        cv2.line(frame, (center_x, bar_y1 - 2), (center_x, bar_y2 + 2), (0, 0, 255), 1)

        # 3. Render 8x8 Board Squares
        board_offset_y = self.header_height + self.eval_bar_height
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                is_light = (file + rank) % 2 == 0
                sq_color = COLOR_LIGHT if is_light else COLOR_DARK

                # Highlight squares from the last move
                if last_move and sq in [last_move.from_square, last_move.to_square]:
                    sq_color = COLOR_HIGHLIGHT

                x1 = file * self.sq_size
                y1 = board_offset_y + rank * self.sq_size
                x2 = x1 + self.sq_size
                y2 = y1 + self.sq_size

                cv2.rectangle(frame, (x1, y1), (x2, y2), sq_color, -1)

                # Render Piece
                piece = board.piece_at(sq)
                if piece:
                    self._draw_piece(frame, piece, x1, y1, self.sq_size)

        self.frames.append(frame)

    def _draw_piece(self, img, piece: chess.Piece, x: int, y: int, size: int):
        """Draws a clean piece representation on the square."""
        center = (x + size // 2, y + size // 2)
        radius = int(size * 0.38)
        
        # Disc background
        disc_color = (250, 250, 250) if piece.color == chess.WHITE else (45, 45, 45)
        border_color = (15, 15, 15) if piece.color == chess.WHITE else (210, 210, 210)
        cv2.circle(img, center, radius, disc_color, -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, border_color, 2, cv2.LINE_AA)

        # Letter glyph inside disc
        symbol = piece.symbol().upper()
        font_scale = 0.8
        thickness = 2
        (w, h), _ = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_pos = (center[0] - w // 2, center[1] + h // 2)
        text_color = (20, 20, 20) if piece.color == chess.WHITE else (240, 240, 240)
        cv2.putText(img, symbol, text_pos, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    def save_mp4(self, filename: str) -> str:
        """Encodes captured frames to MP4 and returns the output path."""
        if not self.recording_active or len(self.frames) == 0:
            return ""

        file_path = os.path.join(self.output_dir, filename)
        if not file_path.endswith(".mp4"):
            file_path += ".mp4"

        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(frame)
            
        # Hold last frame for 2 seconds (4 extra frames at 2 fps)
        for _ in range(int(self.fps * 2)):
            out.write(self.frames[-1])

        out.release()
        self.frames = []
        self.recording_active = False

        # Zero-Bloat Auto Pruning: Keep only latest `max_videos` to preserve disk space
        try:
            existing_videos = [
                os.path.join(self.output_dir, f)
                for f in os.listdir(self.output_dir)
                if f.endswith(".mp4")
            ]
            if len(existing_videos) > self.max_videos:
                existing_videos.sort(key=os.path.getmtime)
                for old_vid in existing_videos[:-self.max_videos]:
                    os.remove(old_vid)
        except Exception:
            pass

        return file_path
