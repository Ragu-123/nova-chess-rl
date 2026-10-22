"""NovaZero Training Pipeline with Multi-GPU Distributed Self-Play and RGPO Loss."""

import os
import random
import copy
import concurrent.futures
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import chess
from tqdm import tqdm

from .constants import move_to_action, action_to_move, ACTION_SPACE_SIZE
from .encoder import encode_board, get_legal_action_mask
from .model import NovaZeroNet
from .mcts import BQSMCTS
from .elo_league import PredecessorLeague, CheckpointRecord
from .video_recorder import ChessVideoRecorder
from .triton_rgpo import compute_rgpo_loss

class ReplayBuffer:
    """Experience replay buffer for storing self-play game trajectories."""
    def __init__(self, capacity: int = 25000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, pi: np.ndarray, value: float):
        self.buffer.append((state, pi, value))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, pis, values = zip(*batch)
        
        return (
            torch.from_numpy(np.array(states, dtype=np.float32)),
            torch.from_numpy(np.array(pis, dtype=np.float32)),
            torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class NovaTrainer:
    """Orchestrates Multi-GPU Parallel Self-Play, League Management, and Network Optimization."""
    def __init__(
        self,
        model: NovaZeroNet,
        devices: list[torch.device] | torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        regret_weight: float = 0.5,
        video_record_prob: float = 0.25,
        output_dir: str = "outputs",
    ):
        if isinstance(devices, (list, tuple)):
            self.devices = list(devices)
        else:
            self.devices = [devices]

        self.primary_device = self.devices[0]
        self.device = self.primary_device
        self.model = model.to(self.primary_device)
        self.regret_weight = regret_weight
        self.video_record_prob = video_record_prob
        self.output_dir = output_dir

        # Initialize dedicated worker models per GPU
        self.worker_models = {}
        for d in self.devices:
            if d == self.primary_device:
                self.worker_models[d] = self.model
            else:
                replica = NovaZeroNet().to(d)
                replica.load_state_dict(self.model.state_dict())
                replica.eval()
                self.worker_models[d] = replica

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.replay_buffer = ReplayBuffer(capacity=30000)
        self.league = PredecessorLeague(max_capacity=10)
        self.start_episode = 1
        self.best_elo = 1200.0
        self.recorder = ChessVideoRecorder(
            output_dir=os.path.join(output_dir, "videos"),
            fps=2,
            square_size=64,
        )

        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str):
        """Resumes model, optimizer, episode count, and league state without memory bloat."""
        if not os.path.exists(checkpoint_path):
            print(f"[Resume] Checkpoint not found at '{checkpoint_path}'. Starting from Episode 1.")
            return

        print(f"[Resume] Loading state from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=self.primary_device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        self.start_episode = checkpoint.get("episode", 0) + 1
        self.league.active_elo = checkpoint.get("league_elo", 1200.0)
        self.best_elo = checkpoint.get("best_elo", self.league.active_elo)

        # Sync worker replicas
        for d, worker_model in self.worker_models.items():
            if d != self.primary_device:
                worker_model.load_state_dict(self.model.state_dict())

        # Restore league pool
        if "league_checkpoints" in checkpoint:
            self.league.checkpoints = []
            for cp in checkpoint["league_checkpoints"]:
                self.league.checkpoints.append(
                    CheckpointRecord(step=cp["step"], weights=cp["weights"], elo=cp["elo"])
                )

        print(
            f"[Resume] Successfully resumed! Next Episode: {self.start_episode} | "
            f"Active Elo: {self.league.active_elo:.1f} | Best Elo: {self.best_elo:.1f} | "
            f"League Pool: {len(self.league.checkpoints)} opponents"
        )

    def self_play_episode(
        self,
        episode_idx: int,
        simulations: int = 40,
        target_device: torch.device = None
    ) -> tuple[list, float, int, str, int, CheckpointRecord]:
        """Plays one game of self-play or match on target GPU and returns trajectory data."""
        if target_device is None:
            target_device = self.primary_device

        board = chess.Board()
        trajectory = []  # (state, pi, active_player)
        step = 0
        
        # Decide if this episode will be recorded (only worker on primary_device records to avoid thread collision)
        can_record = (target_device == self.primary_device)
        recording = self.recorder.maybe_start_recording(episode_idx, probability=self.video_record_prob) if can_record else False

        # Sample opponent from predecessor league
        opp_idx, opp_record = self.league.sample_opponent(temperature=80.0)
        opp_model = None
        if opp_record is not None:
            opp_model = NovaZeroNet().to(target_device)
            opp_model.load_state_dict(opp_record.weights)
            opp_model.eval()

        active_net = self.worker_models[target_device]
        mcts_active = BQSMCTS(active_net, target_device, num_simulations=simulations)
        mcts_opp = BQSMCTS(opp_model if opp_model else active_net, target_device, num_simulations=simulations)

        last_move = None
        eval_score = 0.0

        while not board.is_game_over() and step < 80:
            current_turn = board.turn
            state_tensor = encode_board(board)

            # Determine which MCTS to use
            if opp_model is not None and current_turn == chess.BLACK:
                mcts = mcts_opp
            else:
                mcts = mcts_active

            # Temperature schedule: explore early, exploit later
            temp = 1.0 if step < 12 else 0.2
            pi, eval_score = mcts.search(board, add_noise=True)

            # Store experience for the active agent only
            if opp_model is None or current_turn == chess.WHITE:
                trajectory.append((state_tensor, pi, current_turn))

            # Select Action
            if temp == 1.0:
                action = np.random.choice(ACTION_SPACE_SIZE, p=pi)
            else:
                action = int(np.argmax(pi))

            move = action_to_move(action, board)
            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            if recording:
                self.recorder.capture_frame(board, last_move=last_move, step=step, eval_score=eval_score)

            board.push(move)
            last_move = move
            step += 1

        # Outcome
        if board.is_checkmate():
            white_reward = -1.0 if board.turn == chess.WHITE else 1.0
        else:
            white_reward = 0.0

        video_path = ""
        if recording:
            self.recorder.capture_frame(board, last_move=last_move, step=step, eval_score=eval_score)
            video_path = self.recorder.save_mp4(f"episode_{episode_idx:04d}.mp4")

        return trajectory, white_reward, step, video_path, opp_idx, opp_record

    def train_step(self, batch_size: int = 64) -> dict:
        """Runs a single gradient update across all GPUs using Regret-Gated Policy Optimization."""
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "regret_penalty": 0.0}

        states, target_pis, target_values = self.replay_buffer.sample(batch_size)
        states = states.to(self.primary_device)
        target_pis = target_pis.to(self.primary_device)
        target_values = target_values.to(self.primary_device)

        self.model.train()
        self.optimizer.zero_grad()

        # Multi-GPU batch parallelism if multiple GPUs are available
        if len(self.devices) > 1 and states.is_cuda:
            gpu_ids = [d.index for d in self.devices]
            policy_logits, predicted_values, predicted_regrets = torch.nn.parallel.data_parallel(
                self.model, states, device_ids=gpu_ids
            )
        else:
            policy_logits, predicted_values, predicted_regrets = self.model(states)

        # 1. Value MSE Loss
        loss_val = F.mse_loss(predicted_values, target_values)

        # 2. Fused Regret-Gated Policy Loss (Triton GPU Kernel or PyTorch fallback)
        loss_rgpo = compute_rgpo_loss(policy_logits, target_pis, predicted_regrets, self.regret_weight)

        # Total Loss
        total_loss = loss_val + loss_rgpo
        total_loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "value_loss": loss_val.item(),
            "policy_loss": loss_rgpo.item(),
            "regret_penalty": 0.0,
        }

    def train(self, num_episodes: int = 50, batch_size: int = 64, sim_count: int = 30, checkpoint_freq: int = 10):
        """Runs parallel multi-GPU self-play training with zero disk bloat."""
        target_episodes = self.start_episode + num_episodes - 1
        num_gpus = len(self.devices)
        
        print(f"=== Starting NovaZero Training Pipeline ===")
        print(f"Episodes: {self.start_episode} -> {target_episodes} (Total this run: {num_episodes})")
        print(f"Active GPUs: {num_gpus} {self.devices} | Initial League Elo: {self.league.active_elo:.1f} | Peak Elo: {self.best_elo:.1f}")

        # Register initial weights if league is empty
        if len(self.league.checkpoints) == 0:
            self.league.register_checkpoint(step=self.start_episode - 1, model=self.model, initial_elo=self.league.active_elo)

        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ep = self.start_episode
        while ep <= target_episodes:
            # Batch up to num_gpus episodes to execute simultaneously across all GPUs
            batch_size_eps = min(num_gpus, target_episodes - ep + 1)
            batch_ep_indices = list(range(ep, ep + batch_size_eps))

            # Sync worker models with primary weights
            if num_gpus > 1:
                with torch.no_grad():
                    primary_weights = self.model.state_dict()
                    for d in self.devices:
                        if d != self.primary_device:
                            self.worker_models[d].load_state_dict(primary_weights)

            # Parallel Self-Play across all GPUs
            if num_gpus > 1 and batch_size_eps > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size_eps) as executor:
                    futures = [
                        executor.submit(self.self_play_episode, ep_idx, sim_count, self.devices[i])
                        for i, ep_idx in enumerate(batch_ep_indices)
                    ]
                    results = [f.result() for f in futures]
            else:
                # Single device execution
                results = [self.self_play_episode(batch_ep_indices[0], sim_count, self.primary_device)]

            # Ingest all game results into ReplayBuffer & update Elo
            for i, (trajectory, white_reward, steps, video_path, opp_idx, opp_record) in enumerate(results):
                curr_ep = batch_ep_indices[i]
                for state, pi, player in trajectory:
                    target_value = white_reward if player == chess.WHITE else -white_reward
                    self.replay_buffer.push(state, pi, target_value)

                # Update Elo
                match_score = (white_reward + 1.0) / 2.0
                if opp_record is not None:
                    self.league.update_match_elo(opp_idx, match_score)

                # Gradient Update
                train_metrics = {"loss": 0.0}
                if len(self.replay_buffer) >= batch_size:
                    for _ in range(2):
                        train_metrics = self.train_step(batch_size=batch_size)

                # Progress Logging
                video_flag = f" [MP4 Saved: {os.path.basename(video_path)}]" if video_path else ""
                assigned_gpu = f"[GPU {self.devices[i % num_gpus].index}]" if num_gpus > 1 else ""
                print(
                    f"Episode {curr_ep:04d}/{target_episodes:04d} {assigned_gpu} | Steps: {steps:02d} | "
                    f"Reward: {white_reward:+.1f} | Buffer: {len(self.replay_buffer):04d} | "
                    f"Loss: {train_metrics['loss']:.4f} | Elo: {self.league.active_elo:.1f}{video_flag}"
                )

                # Checkpointing (Strict zero-bloat: last.pt & best.pt only)
                if curr_ep % checkpoint_freq == 0 or curr_ep == target_episodes:
                    last_path = os.path.join(ckpt_dir, "last.pt")
                    best_path = os.path.join(ckpt_dir, "best.pt")

                    league_cps = [
                        {"step": cp.step, "weights": cp.weights, "elo": cp.elo}
                        for cp in self.league.checkpoints
                    ]
                    state = {
                        "episode": curr_ep,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "league_elo": self.league.active_elo,
                        "best_elo": self.best_elo,
                        "league_checkpoints": league_cps,
                    }
                    torch.save(state, last_path)
                    self.league.register_checkpoint(step=curr_ep, model=self.model, initial_elo=self.league.active_elo)
                    print(f"--> [Checkpoint] Saved '{last_path}' (Episode {curr_ep})")

                    if self.league.active_elo > self.best_elo:
                        self.best_elo = self.league.active_elo
                        state["best_elo"] = self.best_elo
                        torch.save(state, best_path)
                        print(f"--> [NEW BEST] Saved '{best_path}' (Peak Elo: {self.best_elo:.1f})")

            ep += batch_size_eps
