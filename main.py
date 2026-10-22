"""Main CLI entry point for NovaZero Chess Reinforcement Learning Engine.

Supported Modes:
- train: Multi-GPU parallel self-play training with resume & zero-bloat checkpointing
- eval: Benchmark tournament against greedy heuristic baseline
- uci: Universal Chess Interface engine protocol (for lichess-bot & cutechess-cli)
- lichess: Standalone live bot runner connected to Lichess Bot API
"""

import os
import argparse
import sys
import torch

from nova_zero.model import NovaZeroNet
from nova_zero.train import NovaTrainer
from nova_zero.eval import run_tournament
from nova_zero.uci import UCIEngine
from nova_zero.lichess_client import LichessBot

def parse_args():
    parser = argparse.ArgumentParser(description="NovaZero: Regret-Gated Reinforcement Learning Chess Engine")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "uci", "lichess"], default="train",
                        help="Execution mode: train, eval, uci, or lichess")
    parser.add_argument("--episodes", type=int, default=30, help="Number of self-play episodes to run this session")
    parser.add_argument("--simulations", type=int, default=35, help="Number of MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training updates")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--regret-weight", type=float, default=0.5, help="Weight for regret-gated policy penalty")
    parser.add_argument("--record-prob", type=float, default=0.25, help="Probability of recording episode as MP4")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific path to checkpoint .pt file")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt without losing progress")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for checkpoints/videos")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto (detects all GPUs), cuda, or cpu")
    parser.add_argument("--lichess-token", type=str, default=None, help="API token for lichess.org BOT account")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine devices (Auto-detect all available GPUs, e.g. 4x L4 or 2x T4)
    if args.device == "auto":
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
            primary_device = devices[0]
            gpu_desc = f"{num_gpus}x GPUs Parallel ({torch.cuda.get_device_name(0)})"
        elif num_gpus == 1:
            devices = [torch.device("cuda:0")]
            primary_device = devices[0]
            gpu_desc = f"1x GPU ({torch.cuda.get_device_name(0)})"
        else:
            devices = [torch.device("cpu")]
            primary_device = devices[0]
            gpu_desc = "CPU"
    else:
        primary_device = torch.device(args.device)
        devices = [primary_device]
        gpu_desc = str(args.device)

    print("=" * 68)
    print("   NOVAZERO: Regret-Gated Elo-League Self-Play Chess Engine")
    print(f"   Target Device: {gpu_desc} | Execution Mode: {args.mode.upper()}")
    if len(devices) > 1:
        print(f"   🚀 Multi-GPU Auto-Scaling ACTIVE across {len(devices)} GPUs in parallel:")
        for i, d in enumerate(devices):
            vram = torch.cuda.get_device_properties(d).total_memory / (1024**3)
            print(f"      - Worker GPU {i}: {torch.cuda.get_device_name(d)} ({vram:.1f} GB VRAM)")
    print("=" * 68)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    best_ckpt = os.path.join(ckpt_dir, "best.pt")

    # 1. UCI Engine Mode
    if args.mode == "uci":
        ckpt_to_load = args.checkpoint or (best_ckpt if os.path.exists(best_ckpt) else (last_ckpt if os.path.exists(last_ckpt) else None))
        engine = UCIEngine(checkpoint_path=ckpt_to_load, device_str=str(primary_device), simulations=args.simulations)
        engine.run()
        return

    # 2. Lichess Live Bot Mode
    if args.mode == "lichess":
        token = args.lichess_token or os.environ.get("LICHESS_BOT_TOKEN")
        if not token:
            print("Error: Lichess Bot API token required! Pass --lichess-token <TOKEN> or set LICHESS_BOT_TOKEN env var.")
            sys.exit(1)
        ckpt_to_load = args.checkpoint or (best_ckpt if os.path.exists(best_ckpt) else (last_ckpt if os.path.exists(last_ckpt) else None))
        bot = LichessBot(token=token, checkpoint_path=ckpt_to_load, simulations=args.simulations, device_str=str(primary_device))
        bot.start_event_stream()
        return

    # 3. Training Mode (Multi-GPU Parallel)
    if args.mode == "train":
        model = NovaZeroNet().to(primary_device)
        trainer = NovaTrainer(
            model=model,
            devices=devices,
            lr=args.lr,
            regret_weight=args.regret_weight,
            video_record_prob=args.record_prob,
            output_dir=args.output_dir,
        )

        # Check for resume
        if args.resume or (args.checkpoint and os.path.exists(args.checkpoint)):
            resume_path = args.checkpoint if (args.checkpoint and os.path.exists(args.checkpoint)) else last_ckpt
            trainer.load_checkpoint(resume_path)

        trainer.train(
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            sim_count=args.simulations,
            checkpoint_freq=max(1, min(10, args.episodes // 2)),
        )
        return

    # 4. Evaluation Mode
    if args.mode == "eval":
        model = NovaZeroNet().to(primary_device)
        ckpt_to_load = args.checkpoint or (best_ckpt if os.path.exists(best_ckpt) else (last_ckpt if os.path.exists(last_ckpt) else None))
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            print(f"Loading checkpoint for evaluation: {ckpt_to_load}")
            checkpoint_data = torch.load(ckpt_to_load, map_location=primary_device)
            model.load_state_dict(checkpoint_data.get("model_state_dict", checkpoint_data))
        else:
            print("Warning: Running tournament with untrained weights (no checkpoint found).")

        run_tournament(
            model=model,
            device=primary_device,
            num_games=10,
            simulations=args.simulations,
            record_video=True,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()
