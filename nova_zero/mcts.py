"""Dual-Tier Bitboard Quiescence MCTS (BQS-MCTS) for NovaZero."""

import math
import numpy as np
import torch
import chess
from .constants import move_to_action, action_to_move, ACTION_SPACE_SIZE
from .encoder import encode_board, get_legal_action_mask

class MCTSNode:
    """Node in the Monte Carlo Search Tree."""
    __slots__ = ['parent', 'children', 'visit_count', 'total_value', 'prior', 'action', 'is_expanded']

    def __init__(self, parent=None, prior: float = 0.0, action: int = -1):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.action = action
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def best_child(self, c_puct: float = 1.4) -> tuple:
        """Selects child with maximum PUCT score."""
        sqrt_total = math.sqrt(self.visit_count)
        best_score = -float('inf')
        best_action = -1
        best_node = None

        for action, child in self.children.items():
            u = c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child

        return best_action, best_node

    def expand(self, action_priors: dict):
        """Expands node with legal action priors."""
        self.is_expanded = True
        for action, prior in action_priors.items():
            self.children[action] = MCTSNode(parent=self, prior=prior, action=action)

    def update(self, value: float):
        """Backpropagates value up the tree."""
        self.visit_count += 1
        self.total_value += value


class BQSMCTS:
    """Monte Carlo Tree Search with Bitboard Tactical Quiescence Rollouts."""
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 50,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board: chess.Board, add_noise: bool = True) -> tuple[np.ndarray, float]:
        """Runs MCTS simulations from the current board position.
        
        Returns:
            pi: Policy probability distribution over 4096 actions.
            root_q: Expected root value.
        """
        root = MCTSNode()
        
        # Initial expansion of root
        root_priors, root_val = self._evaluate_and_expand(root, board)
        
        if add_noise and len(root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]

        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy(stack=False)
            search_path = [node]

            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                action, node = node.best_child(self.c_puct)
                search_path.append(node)
                move = action_to_move(action, scratch_board)
                if move in scratch_board.legal_moves:
                    scratch_board.push(move)
                else:
                    # Invalidation guard
                    break

            # 2. Evaluation & Quiescence Check
            if scratch_board.is_game_over():
                outcome = scratch_board.outcome()
                if outcome.winner is None:
                    value = 0.0
                else:
                    # Positive if current scratch_board active turn won (impossible at game over, so negative)
                    value = -1.0 if outcome.winner != scratch_board.turn else 1.0
            else:
                _, value = self._evaluate_and_expand(node, scratch_board)
                # Apply Tactical Quiescence adjustment if board is in a tactical flare
                value = self._quiescence_check(scratch_board, value)

            # 3. Backpropagation (Value perspective flips each turn)
            for path_node in reversed(search_path):
                path_node.update(value)
                value = -value

        # Calculate final visit distribution
        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            for action, child in root.children.items():
                pi[action] = child.visit_count / total_visits
        else:
            # Fallback to legal uniform
            mask = get_legal_action_mask(board)
            if mask.sum() > 0:
                pi[mask] = 1.0 / mask.sum()

        return pi, root.q_value

    def _evaluate_and_expand(self, node: MCTSNode, board: chess.Board) -> tuple[dict, float]:
        """Runs neural network inference and expands node."""
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value, _ = self.model(tensor)
            
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value = value.item()

        # Mask illegal actions
        legal_mask = get_legal_action_mask(board)
        policy_logits[~legal_mask] = -1e9
        
        # Stable Softmax
        exp_logits = np.exp(policy_logits - np.max(policy_logits[legal_mask]))
        exp_logits[~legal_mask] = 0.0
        sum_exp = exp_logits.sum()
        
        priors = exp_logits / sum_exp if sum_exp > 0 else np.ones_like(exp_logits) / len(exp_logits)
        action_priors = {action: float(priors[action]) for action in np.where(legal_mask)[0]}
        
        node.expand(action_priors)
        return action_priors, value

    def _quiescence_check(self, board: chess.Board, static_val: float) -> float:
        """Fast bitboard tactical quiescence filter.
        
        If immediate hanging captures exist, discounts overly optimistic static evals.
        """
        # Fast capture check: does opponent have high-value captures?
        for move in board.legal_moves:
            if board.is_capture(move):
                piece_captured = board.piece_at(move.to_square)
                if piece_captured and piece_captured.piece_type in [chess.QUEEN, chess.ROOK]:
                    # Tactical volatility detected: damp static optimism towards draw/risk
                    return static_val * 0.8
        return static_val
