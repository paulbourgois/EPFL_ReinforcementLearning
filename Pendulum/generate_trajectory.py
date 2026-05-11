import gymnasium as gym
import numpy as np
import h5py
from stable_baselines3 import SAC

# ── Load best expert ──────────────────────────────────────────────────────────
model = SAC.load("experts/best_model")
env   = gym.make("Pendulum-v1", g=9.81)

def generate_trajectories(model, env, n_trajectories=50, deterministic=True):
    dataset = []

    for i in range(n_trajectories):
        trajectory = {"obs": [], "actions": [], "rewards": [], "dones": []}
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory["obs"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            obs = next_obs

        # Summary per trajectory
        trajectory["total_return"] = sum(trajectory["rewards"])
        dataset.append(trajectory)

        print(f"Traj {i+1:3d} | return: {trajectory['total_return']:.1f} | "
              f"length: {len(trajectory['obs'])}")

    return dataset

# ── Generate different K sizes ────────────────────────────────────────────────
# This is your core ablation — generate once, slice later
dataset = generate_trajectories(model, env, n_trajectories=50)

mean_return = np.mean([t["total_return"] for t in dataset])
print(f"\nExpert mean return over 50 trajectories: {mean_return:.1f}")

# ── Save to HDF5 ──────────────────────────────────────────────────────────────
with h5py.File("datasets/pendulum_expert_50.h5", "w") as f:
    for i, traj in enumerate(dataset):
        grp = f.create_group(f"traj_{i}")
        grp.create_dataset("obs",     data=np.array(traj["obs"]))
        grp.create_dataset("actions", data=np.array(traj["actions"]))
        grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        grp.create_dataset("dones",   data=np.array(traj["dones"]))

print("Saved to datasets/pendulum_expert_50.h5")