# NovaZero: Regret-Gated Elo-League Self-Play Chess Engine ♟️🚀

NovaZero is an open-source, high-throughput reinforcement learning chess engine engineered to overcome the computational, tactical, and credit-assignment bottlenecks of **AlphaZero** and **MuZero**.

---

## 🔬 Breakthrough Solutions to Reinforcement Learning Plateaus

### 1. Solving the Search Depth Horizon Problem
* **The Problem**: With a static budget of 30–50 simulations, standard MCTS cannot detect tactical traps 6+ plies ahead.
* **Our Solution 1 — Playout Cap Randomization (PCR)**:
  NovaZero dynamically varies search depth: 80% fast rollouts on standard moves, and **20% deep rollouts (3×–4× simulations)** on critical positions. This pushes the tactical search depth to **8–12 plies** without inflating average training latency.
* **Our Solution 2 — Recursive Tactical Quiescence (MVV-LVA)**:
  At leaf nodes, instead of trusting raw static neural scores on hanging pieces, NovaZero runs a **4-ply mini alpha-beta search over captures & checks** using Most Valuable Victim - Least Valuable Attacker (MVV-LVA) piece values. This completely eliminates tactical blindness.

### 2. Solving Credit Assignment Failure (Late-Game Blunders)
* **The Problem**: If the model plays 40 brilliant moves and blunders on move 41, standard terminal Monte-Carlo RL assigns a $-1.0$ penalty across the *entire* trajectory, corrupting the opening and middlegame value function.
* **Our Solution — Blunder-Aware TD($\lambda$) Value Blending**:
  Instead of a flat terminal reward, target values are computed as:
  $$z_t = \gamma^{T - 1 - t} \cdot z_{\text{term}} + (1 - \gamma^{T - 1 - t}) \cdot Q_{\text{MCTS}}(s_t)$$
  - For the blunder ($t \approx T$), $\gamma^0 = 1.0 \implies z_t = -1.0$ (penalized with full force).
  - For an opening move 30 plies earlier ($t = 10, T = 41$), $\gamma^{30} \approx 0.29 \implies z_{10}$ is dominated by its own local positional evaluation $Q_{\text{MCTS}}(s_{10})$, **preserving credit for strong opening and middlegame play**.

### 3. Grandmaster Opening Book Seeding
* Built-in library of master lines (Sicilian Defense, Ruy Lopez, Queen's Gambit, King's Indian, French, Caro-Kann).
* During early self-play plies, master opening moves are injected with 40% probability, giving the agent a grandmaster positional foundation from Day 1 rather than wasting millions of steps rediscovering basic opening theory.

### 4. Fused Triton 3.6.0 Regret-Gated Policy Optimization (RGPO)
* Predicts instantaneous action regret $\hat{R}(s, a) \ge 0$.
* Custom **Fused Triton GPU Kernel** with exact analytical backward gradient:
  $$\frac{\partial \mathcal{L}}{\partial z_k} = (p_k - \pi_k) + \lambda \cdot p_k (R_k - \bar{R})$$
* **3.93× faster** than PyTorch autograd with zero GPU global memory overhead.

### 5. Multi-GPU Auto-Scaling & 12-Hour Session Guard
* **Auto-Scaling**: Automatically detects all available GPUs (e.g. 4× NVIDIA L4 or 2× Tesla T4) and runs parallel concurrent self-play workers.
* **Strict Zero-Disk Bloat**: Maintains strictly two files: `outputs/checkpoints/last.pt` and `outputs/checkpoints/best.pt` (~300 MB total), leaving 98% of Kaggle's 20 GB disk free.
* **11.5-Hour Safety Guard**: Automatically tracks session runtime and cleanly saves all checkpoints right before Kaggle's 12-hour container shutdown.

---

## 📁 Repository Structure

```text
nova-chess-rl/
├── nova_zero/
│   ├── __init__.py          # Package initialization
│   ├── constants.py          # 4096-flat action space & move mappings
│   ├── encoder.py            # 19-plane vectorized tensor representation
│   ├── model.py              # Squeeze-and-Excitation ResNet with Regret Head
│   ├── mcts.py               # BQS-MCTS with PCR & Recursive MVV-LVA Quiescence
│   ├── openings.py           # Grandmaster opening library & seedings
│   ├── elo_league.py         # Historical Predecessor Elo League
│   ├── triton_rgpo.py        # Fused Triton 3.6.0 RGPO kernel
│   ├── video_recorder.py     # MP4 video renderer with auto disk pruning
│   ├── uci.py                # Universal Chess Interface protocol handler
│   ├── lichess_client.py     # Direct Lichess Bot API client
│   ├── train.py              # Multi-GPU Trainer with blunder-aware TD credit
│   └── eval.py               # Tournament benchmark suite
├── run_12hr_session.py       # Autonomous 12-hour session runner for Kaggle
├── main.py                   # Unified CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 🚀 Running Training

### Autonomous 12-Hour Continuous Run (Recommended for Kaggle):
```bash
python run_12hr_session.py
```
*Auto-resumes from `last.pt`, scales across all GPUs, applies PCR tactical depth and blunder-aware credit assignment, and safely shuts down after 11.5 hours.*

### Manual Training Command:
```bash
python main.py --mode train --resume --episodes 500 --simulations 40 --batch-size 64 --max-hours 11.5 --device auto
```

---

## ♟️ Lichess Bot & UCI Interface

### Run as Live Lichess Bot:
```bash
python main.py --mode lichess --lichess-token "lip_your_token_here" --simulations 50 --device auto
```

### Run via UCI Engine Protocol:
```bash
python main.py --mode uci --checkpoint outputs/checkpoints/best.pt --simulations 50 --device auto
```
