"""Autonomous 12-Hour Long Training Runner for Kaggle and Cloud GPU Clusters.

Features:
- Auto-detects all GPUs (e.g., 4x NVIDIA L4, 2x Tesla T4)
- Seamless resumption from last.pt without losing training steps or Elo
- Dynamic Playout Cap Randomization (PCR) & Deep Tactical Quiescence (MVV-LVA)
- Blunder-aware credit assignment (TD-blended targets)
- Grandmaster opening seeding to prime professional opening principles
- Strict Zero-Disk Bloat: saves only last.pt and best.pt (~300 MB)
- 11.5-Hour Safety Guard: safely saves checkpoints and gracefully exits before Kaggle's 12-hour hard limit
"""

import sys
import subprocess

def run():
    print("=" * 70)
    print("   NOVAZERO: HIGH-PERFORMANCE 12-HOUR SESSION RUNNER (4x L4 SATURATOR)")
    print("=" * 70)

    # Any extra CLI args passed to run_12hr_session.py are passed through to main.py
    extra_args = sys.argv[1:]

    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--resume",                  # Auto-resumes from last.pt without losing steps/Elo
        "--episodes", "5000",        # Continuous 12h run target
        "--simulations", "150",      # Deep search (Dynamic PCR boosts to 450 on tactics)
        "--workers-per-gpu", "2",    # 2 concurrent games per GPU (8 total parallel games on 4x L4)
        "--batch-size", "256",       # Saturates Tensor Cores across 4 GPUs (64 per GPU)
        "--num-blocks", "10",        # Deep ResNet backbone
        "--num-filters", "192",      # High-capacity feature maps
        "--max-hours", "11.5",       # Exit safely before Kaggle 12h limit
        "--output-dir", "outputs",
        "--device", "auto",          # Parallel auto-scaling across all 4 GPUs
    ] + extra_args

    print(f"Executing: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    run()
