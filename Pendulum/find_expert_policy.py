import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

# ── Environment ──────────────────────────────────────────────────────────────
env      = Monitor(gym.make("Pendulum-v1", g=9.81))
# eval_env = Monitor(gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array"))
eval_env = Monitor(gym.make("Pendulum-v1", g=9.81))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
# These hyperparams are tuned for Pendulum in the SB3 Zoo
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    # Core SAC hyperparams
    learning_rate=1e-3,          # higher than default (3e-4) works better here
    buffer_size=200_000,         # replay buffer size
    learning_starts=1_000,       # random exploration steps before training
    batch_size=256,
    tau=0.005,                   # soft update coefficient for target network
    gamma=0.99,                  # discount factor
    train_freq=1,                # update every step
    gradient_steps=1,
    # Entropy
    ent_coef="auto",             # auto-tune entropy temperature α
    target_entropy="auto",       # sets target to -dim(A) = -1 for Pendulum
    # Network
    policy_kwargs=dict(
        net_arch=[256, 256],     # two hidden layers of 256
        log_std_init=-3,         # tighter initial action distribution
    ),
    seed=42,
)

# ── Callbacks ────────────────────────────────────────────────────────────────
# Stop early if we hit -200 mean reward (near-optimal for Pendulum)
stop_cb = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./experts/",
    log_path="./logs/",
    eval_freq=5_000,             # evaluate every 5k steps
    n_eval_episodes=10,
    deterministic=True,
    callback_after_eval=stop_cb,
    verbose=1,
)

# ── Train ────────────────────────────────────────────────────────────────────
model.learn(total_timesteps=150_000, callback=eval_cb, log_interval=10)

# Saves best model automatically via EvalCallback
# Also save final
model.save("experts/sac_pendulum_final")
print("Done. Best model saved to experts/")