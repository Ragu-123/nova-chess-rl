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
    print("   NOVAZERO: AUTONOMOUS 12-HOUR SESSION RUNNER")
    print("=" * 70)

    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--resume",               # Auto-resumes from last.pt if exists
        "--episodes", "5000",     # Target large batch for continuous 12h run
        "--simulations", "40",    # Baseline simulations (PCR will dynamically boost to 120 on tactics)
        "--batch-size", "64",
        "--max-hours", "11.5",    # Exit safely after 11.5 hours before Kaggle shutdown
        "--output-dir", "outputs",
        "--device", "auto",       # Parallel multi-GPU auto-detection
    ]

    print(f"Executing: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    run()
