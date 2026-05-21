import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor

base_dir = Path(__file__).resolve().parent
experts_dir = base_dir / "experts"
logs_dir = base_dir / "logs"
experts_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

# ── Environment ──────────────────────────────────────────────────────────────
env = Monitor(gym.make("CartPole-v1"))
# eval_env = Monitor(gym.make("CartPole-v1", render_mode="rgb_array"))
eval_env = Monitor(gym.make("CartPole-v1"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
# PPO is usually a much stronger and smoother expert on CartPole than DQN.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,  # discount factor
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs=dict(
        net_arch=[256, 256],  # two hidden layers of 256
    ),
    seed=42,
)

# ── Callbacks ────────────────────────────────────────────────────────────────
# Stop early if we hit 475 mean reward (near-optimal for CartPole)
stop_cb = StopTrainingOnRewardThreshold(reward_threshold=475, verbose=1)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(experts_dir),
    log_path=str(logs_dir),
    eval_freq=1_000,  # evaluate every 1k steps
    n_eval_episodes=10,
    deterministic=True,
    callback_after_eval=stop_cb,
    verbose=1,
)

# ── Train ────────────────────────────────────────────────────────────────────
model.learn(total_timesteps=100_000, callback=eval_cb, log_interval=10)

# Saves best model automatically via EvalCallback
# Also save final
model.save(experts_dir / "ppo_cartpole_final")
print(f"Done. Best model saved to {experts_dir}")
