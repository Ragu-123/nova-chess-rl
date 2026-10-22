"""Main CLI entry point for NovaZero Chess Reinforcement Learning Engine."""

import argparse
import sys
import torch

from nova_zero.model import NovaZeroNet
from nova_zero.train import NovaTrainer
from nova_zero.eval import run_tournament

def parse_args():
    parser = argparse.ArgumentParser(description="NovaZero: Regret-Gated Reinforcement Learning Chess Engine")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode: train or eval")
    parser.add_argument("--episodes", type=int, default=30, help="Number of self-play episodes")
    parser.add_argument("--simulations", type=int, default=35, help="Number of MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training updates")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--regret-weight", type=float, default=0.5, help="Weight for regret-gated policy penalty")
    parser.add_argument("--record-prob", type=float, default=0.25, help="Probability of recording episode as MP4")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for checkpoints/videos")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: cuda, cpu, or auto")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("   NOVAZERO: Regret-Gated Elo-League Self-Play Chess Engine")
    print(f"   Target Device: {device} | Execution Mode: {args.mode.upper()}")
    print("=" * 60)

    # Initialize Model
    model = NovaZeroNet().to(device)

    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])

    if args.mode == "train":
        trainer = NovaTrainer(
            model=model,
            device=device,
            lr=args.lr,
            regret_weight=args.regret_weight,
            video_record_prob=args.record_prob,
            output_dir=args.output_dir,
        )
        trainer.train(
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            sim_count=args.simulations,
            checkpoint_freq=max(1, args.episodes // 3),
        )
    elif args.mode == "eval":
        run_tournament(
            model=model,
            device=device,
            num_games=10,
            simulations=args.simulations,
            record_video=True,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()
