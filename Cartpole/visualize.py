import gymnasium as gym
import numpy as np
import h5py
from pathlib import Path
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

display = "VIDEO"  # "TERMINAL" or "PLOT" or "VIDEO" or None

base_dir = Path(__file__).resolve().parent
experts_dir = base_dir / "experts"
datasets_dir = base_dir / "datasets"
videos_dir = base_dir / "videos"
plots_dir = base_dir / "plots"
videos_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

model = PPO.load(experts_dir / "best_model")


def load_trajectories(path, K):
    """Load first K trajectories from the dataset."""
    dataset = []
    with h5py.File(path, "r") as f:
        for i in range(K):
            grp = f[f"traj_{i}"]
            dataset.append(
                {
                    "obs": grp["obs"][:],
                    "actions": grp["actions"][:],
                    "rewards": grp["rewards"][:],
                    "total_return": np.sum(grp["rewards"][:]),
                }
            )
    return dataset


dataset = load_trajectories(datasets_dir / "cartpole_expert_50.h5", K=10)

if display == "TERMINAL":
    print(f"Loaded {len(dataset)} trajectories:")
    returns = [t["total_return"] for t in dataset]
    print(f"Mean:   {np.mean(returns):.1f}")
    print(f"Std:    {np.std(returns):.1f}")
    print(f"Min:    {np.min(returns):.1f}")
    print(f"Max:    {np.max(returns):.1f}")
elif display == "PLOT":
    returns = [t["total_return"] for t in dataset]
    lengths = [len(t["obs"]) for t in dataset]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Return distribution
    axes[0].hist(returns, bins=10, color="steelblue", edgecolor="white")
    axes[0].axvline(
        np.mean(returns),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(returns):.1f}",
    )
    axes[0].set_title("Expert return distribution")
    axes[0].set_xlabel("Episode return")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Return per trajectory (quality check — should be stable)
    axes[1].plot(returns, marker="o", markersize=3, linewidth=1, color="steelblue")
    axes[1].axhline(np.mean(returns), color="red", linestyle="--", label="Mean")
    axes[1].set_title("Return per trajectory")
    axes[1].set_xlabel("Trajectory index")
    axes[1].set_ylabel("Return")
    axes[1].legend()

    plt.tight_layout()  # type: ignore
    plt.savefig(plots_dir / "expert_trajectories_cartpole.png", dpi=150)
    plt.show()
elif display == "VIDEO":
    video_env = RecordVideo(
        gym.make("CartPole-v1", render_mode="rgb_array"),
        video_folder=str(videos_dir),
        episode_trigger=lambda ep: ep < 3,
        name_prefix="cartpole_expert",
    )
    for _ in range(3):  # ← loop 3 episodes
        obs, _ = video_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
    video_env.close()
    print("Video saved to videos/")
