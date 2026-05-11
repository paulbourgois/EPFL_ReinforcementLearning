"""Minimal PPO training script for Gymnasium Acrobot-v1.

This is intentionally written as readable boilerplate rather than a compact
library implementation. It implements PPO directly with PyTorch so you can see
where policy, value, advantage estimation, and the clipped objective all live.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class PPOConfig:
    env_id: str = "Acrobot-v1"
    seed: int = 42
    total_timesteps: int = 300_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    eval_episodes: int = 10
    eval_interval: int = 10_000
    success_pause_seconds: float = 1.0
    save_path: str = "checkpoints/ppo_acrobot.pt"
    log_dir: str = "runs/ppo_acrobot"
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.shared(obs)).squeeze(-1)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.shared(obs)
        logits = self.policy_head(features)
        return torch.argmax(logits, dim=-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.policy_head(features)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, device: torch.device) -> None:
        self.obs = torch.zeros((steps, obs_dim), device=device)
        self.actions = torch.zeros(steps, device=device, dtype=torch.long)
        self.log_probs = torch.zeros(steps, device=device)
        self.rewards = torch.zeros(steps, device=device)
        self.dones = torch.zeros(steps, device=device)
        self.values = torch.zeros(steps, device=device)
        self.advantages = torch.zeros(steps, device=device)
        self.returns = torch.zeros(steps, device=device)

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = torch.zeros((), device=self.rewards.device)

        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate(agent: ActorCritic, config: PPOConfig, device: torch.device) -> float:
    env = make_env(config.env_id, config.seed + 10_000)
    returns = []

    for episode in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.seed + 10_000 + episode)
        done = False
        episode_return = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_deterministic_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            episode_return += float(reward)

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns))


def train(config: PPOConfig) -> None:
    device = torch.device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config.env_id, config.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    writer = SummaryWriter(config.log_dir)
    save_path = Path(config.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=config.seed)
    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_done = torch.zeros((), device=device)

    global_step = 0
    best_eval_return = -float("inf")

    while global_step < config.total_timesteps:
        buffer = RolloutBuffer(config.rollout_steps, obs_dim, device)

        for step in range(config.rollout_steps):
            global_step += 1
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))

            buffer.actions[step] = action.squeeze(0)
            buffer.log_probs[step] = log_prob.squeeze(0)
            buffer.values[step] = value.squeeze(0)

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            buffer.rewards[step] = float(reward)
            next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(float(done), device=device)

            if done:
                obs, _ = env.reset()
                next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze(0)
            buffer.compute_returns_and_advantages(
                next_value=next_value,
                next_done=next_done,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_indices = np.arange(config.rollout_steps)
        clip_fractions = []

        for _ in range(config.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, config.rollout_steps, config.minibatch_size):
                end = start + config.minibatch_size
                mb_idx = torch.as_tensor(batch_indices[start:end], device=device)

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    buffer.obs[mb_idx], buffer.actions[mb_idx]
                )
                log_ratio = new_log_probs - buffer.log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    )

                mb_advantages = advantages[mb_idx]
                policy_loss_unclipped = -mb_advantages * ratio
                policy_loss_clipped = -mb_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

                value_loss = 0.5 * ((new_values - buffer.returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        explained_var = explained_variance(buffer.values.detach(), buffer.returns.detach())
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clip_fraction", np.mean(clip_fractions), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if global_step % config.eval_interval < config.rollout_steps:
            mean_return = evaluate(agent, config, device)
            writer.add_scalar("eval/mean_return", mean_return, global_step)
            print(f"step={global_step:>7} eval_return={mean_return:>8.2f}")

            if mean_return > best_eval_return:
                best_eval_return = mean_return
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "config": vars(config),
                        "global_step": global_step,
                        "best_eval_return": best_eval_return,
                    },
                    save_path,
                )

    env.close()
    writer.close()
    print(f"Best mean eval return: {best_eval_return:.2f}")
    print(f"Saved best checkpoint to: {save_path}")


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    returns_var = torch.var(returns)
    if returns_var == 0:
        return float("nan")
    return float(1 - torch.var(returns - values) / returns_var)


def watch(config: PPOConfig) -> None:
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed, render_mode="human")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)
    agent = ActorCritic(obs_dim, action_dim).to(device)

    checkpoint = torch.load(config.save_path, map_location=device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode = 1

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.get_deterministic_action(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            status = "success" if terminated else "time_limit"
            print(
                f"episode={episode} status={status} "
                f"return={episode_return:.1f} length={episode_length}"
            )
            time.sleep(config.success_pause_seconds)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            episode += 1


def parse_args() -> tuple[PPOConfig, bool]:
    parser = argparse.ArgumentParser(description="Train PPO on Gymnasium Acrobot-v1.")
    parser.add_argument("--watch", action="store_true", help="Render a saved policy instead of training.")
    parser.add_argument("--env-id", type=str, default=PPOConfig.env_id)
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", type=str, default=PPOConfig.device)
    parser.add_argument("--save-path", type=str, default=PPOConfig.save_path)
    parser.add_argument("--log-dir", type=str, default=PPOConfig.log_dir)
    parser.add_argument("--success-pause-seconds", type=float, default=PPOConfig.success_pause_seconds)
    args = parser.parse_args()

    return PPOConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        device=args.device,
        save_path=args.save_path,
        log_dir=args.log_dir,
        success_pause_seconds=args.success_pause_seconds,
    ), args.watch


if __name__ == "__main__":
    cfg, should_watch = parse_args()
    if should_watch:
        watch(cfg)
    else:
        train(cfg)
