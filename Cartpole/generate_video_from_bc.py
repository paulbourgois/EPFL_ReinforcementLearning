from __future__ import annotations

from pathlib import Path
import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import torch

from ImitationLearning.bc import load_checkpoint, normalize_observations


def record_videos(
    checkpoint_path: Path, video_folder: Path, n_episodes: int = 3, seed: int = 0
) -> None:
    video_folder.mkdir(parents=True, exist_ok=True)

    policy, checkpoint = load_checkpoint(checkpoint_path)
    env_id = str(checkpoint["env_id"])
    obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)

    raw_env = gym.make(env_id, render_mode="rgb_array")
    video_env = RecordVideo(
        raw_env,
        video_folder=str(video_folder),
        episode_trigger=lambda ep: ep < n_episodes,
        name_prefix=f"bc-{checkpoint_path.stem}-{raw_env.spec.id if raw_env.spec else 'env'}",
    )

    policy.eval()
    for ep in range(n_episodes):
        obs, _ = video_env.reset(seed=seed + ep)
        done = False
        total_return = 0.0
        while not done:
            normalized = normalize_observations(
                np.asarray(obs, dtype=np.float32)[None, :], obs_mean, obs_std
            )
            obs_tensor = torch.as_tensor(normalized, dtype=torch.float32)
            with torch.no_grad():
                action_tensor = policy.act(obs_tensor)

            if isinstance(video_env.action_space, gym.spaces.Discrete):
                action = int(action_tensor.item())
            else:
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                action = np.clip(
                    action, video_env.action_space.low, video_env.action_space.high
                )

            obs, reward, terminated, truncated, _ = video_env.step(action)
            total_return += float(reward)
            done = terminated or truncated

        print(f"Recorded episode {ep} (return={total_return:.2f}) -> {video_folder}")

    video_env.close()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record videos from a BC checkpoint (Cartpole)"
    )
    p.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to bc_policy.pt"
    )
    p.add_argument(
        "--n-episodes", type=int, default=3, help="Number of episodes to record"
    )
    p.add_argument(
        "--video-folder",
        type=Path,
        default=Path("Cartpole/il/bc_videos"),
        help="Folder to save videos",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    return p


def main() -> None:
    args = make_parser().parse_args()
    record_videos(args.checkpoint, args.video_folder, args.n_episodes, args.seed)


if __name__ == "__main__":
    main()
