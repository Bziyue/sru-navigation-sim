"""Utilities for parameter-sharing MARL training with RSL-RL."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnv
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.utils.spaces import sample_space


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def parameter_sharing_multi_agent_to_single_agent(env: DirectMARLEnv) -> DirectRLEnv:
    """Flatten agents into the batch dimension so a single shared policy can be trained."""

    class Env(DirectRLEnv):
        def __init__(self, env: DirectMARLEnv) -> None:
            self.env: DirectMARLEnv = env.unwrapped
            self.agent_ids = list(self.env.possible_agents)
            self.original_num_envs = self.env.num_envs
            self.num_agents = len(self.agent_ids)

            self.cfg = self.env.cfg
            self.sim = self.env.sim
            self.scene = self.env.scene
            self.render_mode = self.env.render_mode

            template_agent = self.agent_ids[0]
            self.single_observation_space = gym.spaces.Dict()
            self.single_observation_space["policy"] = self.env.observation_spaces[template_agent]["policy"]
            self.single_observation_space["critic"] = self.env.observation_spaces[template_agent]["critic"]
            self.single_action_space = self.env.action_spaces[template_agent]

            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
            self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

        def __getattr__(self, key: str) -> Any:
            return getattr(self.env, key)

        @property
        def num_envs(self) -> int:
            return self.original_num_envs * self.num_agents

        @property
        def episode_length_buf(self) -> torch.Tensor:
            return torch.cat([self.env.episode_length_buf for _ in self.agent_ids], dim=0)

        @episode_length_buf.setter
        def episode_length_buf(self, value: torch.Tensor):
            if value.numel() == self.original_num_envs:
                self.env.episode_length_buf = value
                return
            reshaped = value.reshape(self.num_agents, self.original_num_envs)
            self.env.episode_length_buf = reshaped[0].clone()

        def _batch_observations(self, obs: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {
                "policy": torch.cat([obs[agent]["policy"] for agent in self.agent_ids], dim=0),
                "critic": torch.cat([obs[agent]["critic"] for agent in self.agent_ids], dim=0),
            }

        def _batch_tensor_dict(self, tensor_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([tensor_dict[agent] for agent in self.agent_ids], dim=0)

        def _combine_extras(self, extras: dict[str, dict]) -> dict:
            if not extras:
                return {}
            combined = {"agent_extras": extras}

            merged_log = {}
            for agent in self.agent_ids:
                agent_log = extras.get(agent, {}).get("log")
                if not isinstance(agent_log, dict):
                    continue
                for key, value in agent_log.items():
                    if key not in merged_log:
                        merged_log[key] = value
                    elif not _values_equal(merged_log[key], value):
                        merged_log[f"{agent}/{key}"] = value

            if merged_log:
                combined["log"] = merged_log
            return combined

        def _get_observations(self) -> dict[str, torch.Tensor]:
            return self._batch_observations(self.env._get_observations())

        def seed(self, seed: int = -1) -> int:
            return self.env.seed(seed)

        def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
            obs, extras = self.env.reset(seed=seed, options=options)
            return self._batch_observations(obs), self._combine_extras(extras)

        def step(self, action: torch.Tensor) -> VecEnvStepReturn:
            split_actions = {}
            for index, agent in enumerate(self.agent_ids):
                start = index * self.original_num_envs
                end = start + self.original_num_envs
                split_actions[agent] = action[start:end]

            obs, rewards, terminated, time_outs, extras = self.env.step(split_actions)
            batched_obs = self._batch_observations(obs)
            batched_rewards = self._batch_tensor_dict(rewards)
            batched_terminated = self._batch_tensor_dict(terminated)
            batched_time_outs = self._batch_tensor_dict(time_outs)
            return batched_obs, batched_rewards, batched_terminated, batched_time_outs, self._combine_extras(extras)

        def render(self, recompute: bool = False) -> np.ndarray | None:
            return self.env.render(recompute)

        def close(self) -> None:
            self.env.close()

    return Env(env)
