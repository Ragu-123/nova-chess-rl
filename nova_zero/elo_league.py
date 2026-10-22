"""Predecessor League: Elo-calibrated historical opponent curriculum."""

import os
import copy
import random
import numpy as np
import torch

class CheckpointRecord:
    def __init__(self, step: int, weights: dict, elo: float = 1200.0):
        self.step = step
        self.weights = weights  # State dict in CPU memory
        self.elo = elo
        self.games_played = 0

class PredecessorLeague:
    """Maintains historical predecessor checkpoints with dynamic Elo calibration."""
    def __init__(self, max_capacity: int = 15, base_elo: float = 1200.0, k_factor: float = 32.0):
        self.max_capacity = max_capacity
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.checkpoints: list[CheckpointRecord] = []
        self.active_elo = base_elo

    def register_checkpoint(self, step: int, model: torch.nn.Module, initial_elo: float = None):
        """Adds a new historical snapshot of model weights to the league."""
        if initial_elo is None:
            initial_elo = self.active_elo

        # Clone weights to CPU
        cpu_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        record = CheckpointRecord(step=step, weights=cpu_weights, elo=initial_elo)
        
        self.checkpoints.append(record)
        if len(self.checkpoints) > self.max_capacity:
            # Drop earliest checkpoint (preserving diverse Elo distribution)
            self.checkpoints.pop(0)

    def sample_opponent(self, temperature: float = 100.0) -> tuple[int, CheckpointRecord]:
        """Samples an opponent using Boltzmann distribution over Elo difference.
        
        Prefers opponents close to the current agent's skill (curriculum learning).
        """
        if not self.checkpoints:
            return -1, None

        elos = np.array([cp.elo for cp in self.checkpoints], dtype=np.float64)
        # Difference from current active elo
        diffs = -(np.abs(self.active_elo - elos)) / max(temperature, 1.0)
        exp_diffs = np.exp(diffs - np.max(diffs))
        probs = exp_diffs / exp_diffs.sum()

        chosen_idx = np.random.choice(len(self.checkpoints), p=probs)
        return chosen_idx, self.checkpoints[chosen_idx]

    def update_match_elo(self, opponent_idx: int, outcome: float):
        """Updates active agent and opponent Elo ratings after a game.
        
        outcome: 1.0 for agent win, 0.5 for draw, 0.0 for opponent win.
        """
        if opponent_idx < 0 or opponent_idx >= len(self.checkpoints):
            return

        opp = self.checkpoints[opponent_idx]
        
        # Expected score for active agent
        ea = 1.0 / (1.0 + 10.0 ** ((opp.elo - self.active_elo) / 400.0))
        eb = 1.0 - ea

        # Update ratings
        self.active_elo += self.k_factor * (outcome - ea)
        opp.elo += self.k_factor * ((1.0 - outcome) - eb)
        opp.games_played += 1
