from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    observations: np.ndarray
    actions: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mode: str
    action_low: np.ndarray | None = None
    action_high: np.ndarray | None = None


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class DiscreteBCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        self.backbone = MLPBackbone(obs_dim, hidden_sizes)
        self.head = nn.Linear(self.backbone.output_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        return self.head(features)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.forward(observations)
        return torch.argmax(logits, dim=-1)


class ContinuousBCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.backbone = MLPBackbone(obs_dim, hidden_sizes)
        self.head = nn.Linear(self.backbone.output_dim, action_dim)
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer(
            "action_bias", (action_high_tensor + action_low_tensor) / 2.0
        )
        self.register_buffer(
            "action_scale", (action_high_tensor - action_low_tensor) / 2.0
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        raw_actions = torch.tanh(self.head(features))
        return self.action_bias + self.action_scale * raw_actions

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward(observations)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sorted_trajectory_keys(keys: Iterable[str]) -> list[str]:
    def sort_key(name: str) -> tuple[int, str]:
        suffix = name.split("_")[-1]
        return (int(suffix) if suffix.isdigit() else 10**9, name)

    return sorted(keys, key=sort_key)


def load_expert_dataset(
    dataset_path: Path,
    max_trajectories: int | None,
    env: gym.Env,
) -> DatasetBundle:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    with h5py.File(dataset_path, "r") as file_handle:
        keys = sorted_trajectory_keys(file_handle.keys())
        if max_trajectories is not None:
            keys = keys[:max_trajectories]

        for key in keys:
            trajectory = file_handle[key]
            observations.append(np.asarray(trajectory["obs"][:], dtype=np.float32))
            actions.append(np.asarray(trajectory["actions"][:]))

    if not observations:
        raise ValueError(f"No trajectories found in {dataset_path}")

    observations_array = np.concatenate(observations, axis=0).astype(np.float32)
    actions_array = np.concatenate(actions, axis=0)

    action_space = env.action_space
    if isinstance(action_space, Discrete):
        actions_array = actions_array.reshape(-1).astype(np.int64)
        action_mode = "discrete"
        action_low = None
        action_high = None
    elif isinstance(action_space, Box):
        actions_array = actions_array.astype(np.float32)
        if actions_array.ndim == 1:
            actions_array = actions_array.reshape(-1, 1)
        action_mode = "continuous"
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported action space: {action_space}")

    obs_mean = observations_array.mean(axis=0, keepdims=True)
    obs_std = observations_array.std(axis=0, keepdims=True)
    obs_std = np.maximum(obs_std, 1e-6)

    return DatasetBundle(
        observations=observations_array,
        actions=actions_array,
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        action_mode=action_mode,
        action_low=action_low,
        action_high=action_high,
    )


def normalize_observations(
    observations: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray
) -> np.ndarray:
    return (observations - obs_mean) / obs_std


def build_policy(
    obs_dim: int,
    action_mode: str,
    action_space: gym.Space,
    hidden_sizes: Iterable[int],
) -> nn.Module:
    if action_mode == "discrete":
        if not isinstance(action_space, Discrete):
            raise TypeError("Discrete action mode requires a Discrete action space")
        return DiscreteBCPolicy(obs_dim, action_space.n, hidden_sizes)

    if not isinstance(action_space, Box):
        raise TypeError("Continuous action mode requires a Box action space")

    return ContinuousBCPolicy(
        obs_dim=obs_dim,
        action_dim=int(np.prod(action_space.shape)),
        hidden_sizes=hidden_sizes,
        action_low=np.asarray(action_space.low, dtype=np.float32),
        action_high=np.asarray(action_space.high, dtype=np.float32),
    )


@torch.no_grad()
def evaluate_policy(
    env_id: str,
    policy: nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    n_episodes: int,
    seed: int,
    render_mode: str | None = None,
) -> dict[str, float]:
    env = gym.make(env_id, render_mode=render_mode)
    policy.eval()
    returns: list[float] = []

    for episode_index in range(n_episodes):
        observations, _ = env.reset(seed=seed + episode_index)
        terminated = truncated = False
        total_return = 0.0

        while not (terminated or truncated):
            normalized = normalize_observations(
                np.asarray(observations, dtype=np.float32)[None, :],
                obs_mean,
                obs_std,
            )
            observation_tensor = torch.as_tensor(normalized, dtype=torch.float32)
            action_tensor = policy.act(observation_tensor)

            if isinstance(env.action_space, Discrete):
                action = int(action_tensor.item())
            else:
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            observations, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)

        returns.append(total_return)

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }


def train_policy(
    env_id: str,
    dataset_path: Path,
    output_dir: Path,
    max_trajectories: int | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    hidden_sizes: tuple[int, ...],
    eval_episodes: int,
) -> dict[str, float]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    dataset = load_expert_dataset(dataset_path, max_trajectories, env)

    observations = normalize_observations(
        dataset.observations, dataset.obs_mean, dataset.obs_std
    )
    actions = dataset.actions

    indices = np.arange(len(observations))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    validation_size = max(1, int(len(indices) * validation_split))
    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]

    train_observations = torch.as_tensor(
        observations[train_indices], dtype=torch.float32
    )
    validation_observations = torch.as_tensor(
        observations[validation_indices], dtype=torch.float32
    )

    if dataset.action_mode == "discrete":
        train_actions = torch.as_tensor(actions[train_indices], dtype=torch.long)
        validation_actions = torch.as_tensor(
            actions[validation_indices], dtype=torch.long
        )
    else:
        train_actions = torch.as_tensor(actions[train_indices], dtype=torch.float32)
        validation_actions = torch.as_tensor(
            actions[validation_indices], dtype=torch.float32
        )

    train_loader = DataLoader(
        TensorDataset(train_observations, train_actions),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    policy = build_policy(
        obs_dim=observations.shape[1],
        action_mode=dataset.action_mode,
        action_space=env.action_space,
        hidden_sizes=hidden_sizes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch_index in range(1, epochs + 1):
        policy.train()
        epoch_losses: list[float] = []

        for batch_observations, batch_actions in train_loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = policy(batch_observations)

            if dataset.action_mode == "discrete":
                loss = F.cross_entropy(outputs, batch_actions)
            else:
                loss = F.mse_loss(outputs, batch_actions)

            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        policy.eval()
        with torch.no_grad():
            validation_observations = validation_observations.to(device)
            validation_actions = validation_actions.to(device)
            validation_outputs = policy(validation_observations)
            if dataset.action_mode == "discrete":
                validation_loss = F.cross_entropy(
                    validation_outputs, validation_actions
                )
            else:
                validation_loss = F.mse_loss(validation_outputs, validation_actions)

        mean_train_loss = float(np.mean(epoch_losses))
        validation_loss_value = float(validation_loss.item())
        history.append(
            {
                "epoch": float(epoch_index),
                "train_loss": mean_train_loss,
                "validation_loss": validation_loss_value,
            }
        )

        if validation_loss_value < best_validation_loss:
            best_validation_loss = validation_loss_value
            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter in policy.state_dict().items()
            }

        print(
            f"Epoch {epoch_index:03d} | train loss {mean_train_loss:.4f} | "
            f"val loss {validation_loss_value:.4f}"
        )

    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)

    checkpoint = {
        "env_id": env_id,
        "action_mode": dataset.action_mode,
        "hidden_sizes": list(hidden_sizes),
        "state_dict": policy.state_dict(),
        "obs_mean": dataset.obs_mean,
        "obs_std": dataset.obs_std,
        "action_low": dataset.action_low,
        "action_high": dataset.action_high,
        "seed": seed,
    }

    checkpoint_path = output_dir / "bc_policy.pt"
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "best_validation_loss": best_validation_loss,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "dataset_size": float(len(observations)),
        "train_size": float(len(train_indices)),
        "validation_size": float(len(validation_indices)),
    }

    evaluation = evaluate_policy(
        env_id=env_id,
        policy=policy,
        obs_mean=dataset.obs_mean,
        obs_std=dataset.obs_std,
        n_episodes=eval_episodes,
        seed=seed + 10_000,
    )
    metrics.update(evaluation)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump({"history": history, "metrics": metrics}, file_handle, indent=2)

    env.close()
    print(f"Saved checkpoint to {checkpoint_path}")
    print(
        f"Evaluation over {eval_episodes} episodes: "
        f"mean={evaluation['mean_return']:.2f}, std={evaluation['std_return']:.2f}"
    )
    return metrics


def load_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    env_id = str(checkpoint["env_id"])
    env = gym.make(env_id)
    policy = build_policy(
        obs_dim=int(np.asarray(checkpoint["obs_mean"]).shape[-1]),
        action_mode=str(checkpoint["action_mode"]),
        action_space=env.action_space,
        hidden_sizes=tuple(int(size) for size in checkpoint["hidden_sizes"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    env.close()
    return policy, checkpoint


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Behavior cloning for expert trajectory datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a BC policy from expert trajectories."
    )
    train_parser.add_argument(
        "--env-id", required=True, help="Gymnasium environment id, e.g. CartPole-v1"
    )
    train_parser.add_argument(
        "--dataset", required=True, type=Path, help="Path to the expert HDF5 dataset"
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the trained policy",
    )
    train_parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit the number of expert trajectories",
    )
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--validation-split", type=float, default=0.2, help="Validation split fraction"
    )
    train_parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,256",
        help="Comma-separated MLP hidden sizes",
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes used for evaluation after training",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved BC checkpoint.")
    eval_parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to bc_policy.pt"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    eval_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    eval_parser.add_argument(
        "--render-mode", type=str, default=None, help="Optional Gymnasium render mode"
    )

    return parser


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "train":
        hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
        train_policy(
            env_id=args.env_id,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_trajectories=args.max_trajectories,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            hidden_sizes=hidden_sizes,
            eval_episodes=args.eval_episodes,
        )
        return

    policy, checkpoint = load_checkpoint(args.checkpoint)
    evaluation = evaluate_policy(
        env_id=str(checkpoint["env_id"]),
        policy=policy,
        obs_mean=np.asarray(checkpoint["obs_mean"], dtype=np.float32),
        obs_std=np.asarray(checkpoint["obs_std"], dtype=np.float32),
        n_episodes=args.episodes,
        seed=args.seed,
        render_mode=args.render_mode,
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
