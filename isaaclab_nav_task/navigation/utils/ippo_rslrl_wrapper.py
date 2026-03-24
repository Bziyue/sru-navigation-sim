"""RSL-RL wrapper that flattens a multi-agent Isaac Lab env into per-agent batches."""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectMARLEnv


class RslRlParameterSharingVecEnvWrapper(VecEnv):
    """Expose a DirectMARLEnv as a parameter-sharing VecEnv for shared-policy PPO/IPPO."""

    def __init__(self, env: DirectMARLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, DirectMARLEnv):
            raise ValueError(f"Expected DirectMARLEnv, got: {type(env)}")

        self.env = env
        self.clip_actions = clip_actions
        self.agent_ids = list(self.env.unwrapped.possible_agents)
        self.base_num_envs = self.env.unwrapped.num_envs
        self.num_agents = len(self.agent_ids)
        self.num_envs = self.base_num_envs * self.num_agents
        self.device = self.env.unwrapped.device
        self.max_episode_length = self.env.unwrapped.max_episode_length
        self.num_actions = gym.spaces.flatdim(self.env.unwrapped.action_spaces[self.agent_ids[0]])
        self.env.reset()

    @property
    def cfg(self):
        return self.env.unwrapped.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf.repeat(self.num_agents)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        if value.numel() == self.base_num_envs:
            self.env.unwrapped.episode_length_buf = value
            return
        reshaped = value.view(self.num_agents, self.base_num_envs)
        self.env.unwrapped.episode_length_buf = reshaped[0].clone()

    def seed(self, seed: int = -1) -> int:
        return self.env.unwrapped.seed(seed)

    def set_training_iteration(self, iteration: int):
        if hasattr(self.env.unwrapped, "set_training_iteration"):
            self.env.unwrapped.set_training_iteration(iteration)

    def initialize_teammate_obs_curriculum(self, anchor_iteration: int | None = None):
        if hasattr(self.env.unwrapped, "initialize_teammate_obs_curriculum"):
            self.env.unwrapped.initialize_teammate_obs_curriculum(anchor_iteration)

    def get_checkpoint_state(self) -> dict | None:
        if hasattr(self.env.unwrapped, "get_checkpoint_state"):
            return self.env.unwrapped.get_checkpoint_state()
        return None

    def load_checkpoint_state(self, state: dict | None):
        if hasattr(self.env.unwrapped, "load_checkpoint_state"):
            self.env.unwrapped.load_checkpoint_state(state)

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        return self._flatten_obs(obs), self._flatten_extras(extras)

    def get_observations(self) -> TensorDict:
        return self._flatten_obs(self.env.unwrapped._get_observations())

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        action_dict = {}
        for agent_idx, agent in enumerate(self.agent_ids):
            start = agent_idx * self.base_num_envs
            end = start + self.base_num_envs
            action_dict[agent] = actions[start:end]

        obs, rewards, terminated, truncated, extras = self.env.step(action_dict)
        flat_obs = self._flatten_obs(obs)
        flat_rewards = torch.cat([rewards[agent] for agent in self.agent_ids], dim=0)
        flat_dones = torch.cat(
            [torch.logical_or(terminated[agent], truncated[agent]).to(dtype=torch.long) for agent in self.agent_ids],
            dim=0,
        )
        flat_extras = self._flatten_extras(extras)
        if not self.env.unwrapped.cfg.is_finite_horizon:
            flat_extras["time_outs"] = torch.cat([truncated[agent] for agent in self.agent_ids], dim=0)
        return flat_obs, flat_rewards, flat_dones, flat_extras

    def close(self):
        return self.env.close()

    def _flatten_obs(self, obs: dict) -> TensorDict:
        first_agent_obs = obs[self.agent_ids[0]]
        if isinstance(first_agent_obs, dict):
            flat_obs = {}
            for key in first_agent_obs:
                flat_obs[key] = torch.cat([obs[agent][key] for agent in self.agent_ids], dim=0)
        else:
            flat_obs = {"policy": torch.cat([obs[agent] for agent in self.agent_ids], dim=0)}
        return TensorDict(flat_obs, batch_size=[self.num_envs])

    def _flatten_extras(self, extras: dict) -> dict:
        if not isinstance(extras, dict) or not extras:
            return {}
        if all(agent in extras for agent in self.agent_ids):
            merged = {}
            log_payload = {}
            log_candidates = [
                extras[agent].get("log")
                for agent in self.agent_ids
                if isinstance(extras[agent], dict) and extras[agent].get("log") is not None
            ]
            metric_candidates = [
                extras[agent].get("metrics")
                for agent in self.agent_ids
                if isinstance(extras[agent], dict) and extras[agent].get("metrics") is not None
            ]
            if metric_candidates:
                log_payload.update(metric_candidates[0])
            if log_candidates:
                log_payload.update(log_candidates[0])
            if log_payload:
                merged["log"] = log_payload
            if "critic" in extras[self.agent_ids[0]]:
                merged["observations"] = {
                    "critic": torch.cat([extras[agent]["critic"] for agent in self.agent_ids], dim=0)
                }
            return merged
        return extras
