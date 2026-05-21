from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bc import parse_hidden_sizes, train_policy


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run behavior cloning sweeps over dataset sizes and seeds.")
    parser.add_argument("--env-id", required=True, help="Gymnasium environment id, e.g. CartPole-v1")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the expert HDF5 dataset")
    parser.add_argument("--output-root", required=True, type=Path, help="Root directory for sweep outputs")
    parser.add_argument("--sizes", type=str, default="1,5,10,25,50", help="Comma-separated number of expert trajectories")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--hidden-sizes", type=str, default="256,256", help="Comma-separated MLP hidden sizes")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes used for evaluation after training")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for size in sizes:
        for seed in seeds:
            output_dir = args.output_root / f"k{size}" / f"seed{seed}"
            metrics = train_policy(
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
