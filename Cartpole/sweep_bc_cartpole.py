from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


def load_cartpole_video_module(repo_root: Path):
    gen_path = repo_root / "Cartpole" / "generate_video_from_bc.py"
    spec = importlib.util.spec_from_file_location("cartpole_gen_video", str(gen_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# Load ImitationLearning/bc.py as a module (avoids package import issues)
def load_bc_module(repo_root: Path):
    bc_path = repo_root / "ImitationLearning" / "bc.py"
    # Use the original module name so dataclasses and annotations resolve correctly
    spec = importlib.util.spec_from_file_location("ImitationLearning.bc", str(bc_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Ensure the module is registered in sys.modules under its spec name
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cartpole BC sweep and video recording"
    )
    parser.add_argument(
        "--env-id", required=True, help="Gymnasium environment id, e.g. CartPole-v1"
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, help="Path to the expert HDF5 dataset"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root directory for sweep outputs",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1,5,10,25,50",
        help="Comma-separated number of expert trajectories",
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2", help="Comma-separated random seeds"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--validation-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,256",
        help="Comma-separated MLP hidden sizes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes used for evaluation after training",
    )
    parser.add_argument(
        "--record-videos",
        action="store_true",
        help="Record BC videos after each experiment",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to record per experiment",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    bc = load_bc_module(repo_root)
    gen_video = load_cartpole_video_module(repo_root)
    record_videos = gen_video.record_videos

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    hidden_sizes = bc.parse_hidden_sizes(args.hidden_sizes)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for size in sizes:
        for seed in seeds:
            output_dir = args.output_root / f"k{size}" / f"seed{seed}"
            metrics = bc.train_policy(
                env_id=args.env_id,
                dataset_path=args.dataset,
                output_dir=output_dir,
                max_trajectories=size,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_split=args.validation_split,
                hidden_sizes=hidden_sizes,
                eval_episodes=args.eval_episodes,
            )

            # Optionally record videos for qualitative comparison
            if args.record_videos:
                checkpoint = output_dir / "bc_policy.pt"
                video_folder = output_dir / "videos"
                if checkpoint.exists():
                    record_videos(
                        checkpoint, video_folder, n_episodes=args.n_episodes, seed=seed
                    )

            row = {"size": size, "seed": seed, **metrics, "output_dir": str(output_dir)}
            rows.append(row)

    csv_path = args.output_root / "results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep results to {csv_path}")


if __name__ == "__main__":
    main()
