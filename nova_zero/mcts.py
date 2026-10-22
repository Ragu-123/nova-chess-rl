"""Dual-Tier Bitboard Quiescence MCTS (BQS-MCTS) for NovaZero."""

import math
import numpy as np
import torch
import chess
import random
from .constants import move_to_action, action_to_move, ACTION_SPACE_SIZE
from .encoder import encode_board, get_legal_action_mask

# Piece material values for tactical quiescence
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

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
        max_tt_size: int = 50000,
        deep_rollout_prob: float = 0.20,
        deep_rollout_mult: float = 3.0,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_tt_size = max_tt_size
        self.deep_rollout_prob = deep_rollout_prob
        self.deep_rollout_mult = deep_rollout_mult
        self.transposition_table = {}  # Zobrist hash -> (action_priors, value)

    def clear_cache(self):
        """Resets the transposition cache."""
        self.transposition_table.clear()

    def search(self, board: chess.Board, add_noise: bool = True, pcr: bool = True) -> tuple[np.ndarray, float]:
        """Runs MCTS simulations with Playout Cap Randomization (PCR) for deep horizon vision.
        
        Returns:
            pi: Policy probability distribution over 4096 actions.
            root_q: Expected root value.
        """
        root = MCTSNode()
        
        # Playout Cap Randomization: 20% deep rollouts (3x sims) to break horizon plateaus
        sims = self.num_simulations
        if pcr and (random.random() < self.deep_rollout_prob):
            sims = int(self.num_simulations * self.deep_rollout_mult)

        # Initial expansion of root
        root_priors, root_val = self._evaluate_and_expand(root, board)
        
        if add_noise and len(root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]

        for _ in range(sims):
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
        """Runs fast neural network inference with transposition table caching."""
        # 1. Transposition Table Lookup (0.01 ms O(1) cache hit)
        zkey = board._transposition_key()
        if zkey in self.transposition_table:
            action_priors, value = self.transposition_table[zkey]
            node.expand(action_priors)
            return action_priors, value

        # 2. Optimized Neural Forward Pass
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device, non_blocking=True)
        self.model.eval()

        with torch.inference_mode():
            if hasattr(self.model, "predict_eval"):
                policy_logits, value = self.model.predict_eval(tensor)
            else:
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
        
        # Store in bounded transposition cache
        if len(self.transposition_table) < self.max_tt_size:
            self.transposition_table[zkey] = (action_priors, value)

        node.expand(action_priors)
        return action_priors, value

    def _quiescence_check(self, board: chess.Board, static_val: float, max_depth: int = 3) -> float:
        """Deep tactical quiescence search on captures/checks (MVV-LVA).
        
        Solves the search depth horizon problem by recursively verifying forcing tactical exchanges.
        """
        stand_pat = self._eval_material_balance(board)
        
        # Fast exit if quiet position
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if not captures or max_depth <= 0:
            return static_val

        # Sort captures by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
        def mvv_lva_score(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
            a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            return v_val * 10 - a_val

        captures.sort(key=mvv_lva_score, reverse=True)

        best_score = stand_pat
        for move in captures[:4]:  # Top 4 most forcing captures
            board.push(move)
            score = -self._quiescence_eval(board, -best_score - 500, -best_score, max_depth - 1)
            board.pop()
            if score > best_score:
                best_score = score

        # Convert tactical delta to [-1, 1] tanh blend
        tactical_val = math.tanh((best_score - stand_pat) / 400.0)
        return float(np.clip(0.60 * static_val + 0.40 * tactical_val, -1.0, 1.0))

    def _quiescence_eval(self, board: chess.Board, alpha: int, beta: int, depth: int) -> int:
        stand_pat = self._eval_material_balance(board)
        if depth <= 0 or board.is_game_over():
            return stand_pat
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        for move in captures[:3]:
            board.push(move)
            score = -self._quiescence_eval(board, -beta, -alpha, depth - 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _eval_material_balance(self, board: chess.Board) -> int:
        """Centipawn score relative to the current player to move."""
        score = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                v = PIECE_VALUES.get(p.piece_type, 0)
                score += v if p.color == board.turn else -v
        return score
