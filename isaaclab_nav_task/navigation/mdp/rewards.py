# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reward functions for navigation tasks.

These functions can be passed to :class:`isaaclab.managers.RewardTermCfg`
to specify the reward function and its parameters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l1(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L1 kernel."""
    return torch.sum(torch.abs(env.action_manager.action - env.action_manager.prev_action), dim=1)


def height_command_l1(env: "ManagerBasedRLEnv", action_name: str = "accel_command", height_index: int = 3) -> torch.Tensor:
    """Penalize aggressive relative height commands."""
    action_term = env.action_manager.get_term(action_name)
    if not getattr(action_term.cfg, "enable_height_command", True):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    processed_actions = action_term.processed_actions
    if processed_actions.shape[1] <= height_index:
        return torch.zeros(env.num_envs, dtype=processed_actions.dtype, device=env.device)
    return torch.abs(processed_actions[:, height_index])


def height_band_penalty(
    env: "ManagerBasedRLEnv",
    cruise_height: float,
    deadband: float = 0.15,
    max_error: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "robot_goal",
    activate_after_start_distance: float = 2.0,
    release_goal_distance: float = 5.0,
) -> torch.Tensor:
    """Penalize height drift only during the mid-course cruise segment.

    The penalty stays inactive near the spawn region and near the goal region so
    the policy can depart and arrive with less vertical restriction. During the
    cruise segment, the vehicle is encouraged to stay near ``cruise_height``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]

    active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    goal_cmd_generator = env.command_manager._terms.get(command_name)
    if goal_cmd_generator is not None:
        spawn_xy = goal_cmd_generator.spawn_position_world[:, :2]
        goal_xy = goal_cmd_generator.goal_position_world[:, :2]
        distance_from_spawn = torch.norm(current_xy - spawn_xy, dim=1)
        distance_to_goal = torch.norm(current_xy - goal_xy, dim=1)
        active_mask = (distance_from_spawn >= float(activate_after_start_distance)) & (
            distance_to_goal >= float(release_goal_distance)
        )

    cruise_heights = torch.full((env.num_envs,), float(cruise_height), dtype=asset.data.root_pos_w.dtype, device=env.device)
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - cruise_heights)
    excess_error = torch.clamp(height_error - deadband, min=0.0)
    return (excess_error / max(max_error, 1e-6)) * active_mask.float()


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving laterally using L1-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.abs(lateral_velocity)
    return reward


def rot_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for rotating around the z-axis using an L2-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the rotational velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rot_vel_norm = torch.norm(asset.data.root_ang_vel_b, dim=1)
    return rot_vel_norm


def reach_goal_xyz(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigmoid: float,
    T_r: float,
    probability: float,
    flat: bool,
    ratio: bool,
) -> torch.Tensor:
    """Reward goal reaching with configurable sigmoid shaping.

    Args:
        env: The learning environment.
        command_name: Name of the goal command.
        sigmoid: Sigmoid parameter for shaping.
        T_r: Time reward scaling factor.
        probability: Probability of random sampling.
        flat: Whether to only consider xy error (ignore z).
        ratio: Whether to scale by travel distance ratio.

    Returns:
        Dense reward based on distance to goal.
    """
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[command_name]
    asset: Articulation = env.scene["robot"]

    t = env.episode_length_buf
    T = env.max_episode_length

    if flat:
        xyz_error = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.goal_position_world[:, :2], dim=1)
    else:
        xyz_error = torch.norm(asset.data.root_pos_w[:, :3] - goal_cmd_generator.goal_position_world[:, :3], dim=1)

    reward = 1 / (1 + torch.square(xyz_error / sigmoid)) / T_r

    timeup_mask = t > (T - goal_cmd_generator.required_time_at_goal_in_steps)
    random_mask = torch.rand_like(t.float()) < probability
    timeup_mask = torch.logical_or(timeup_mask, random_mask)

    arrive_mask = goal_cmd_generator.time_at_goal > 0.0
    reward_mask = torch.logical_or(timeup_mask, arrive_mask)

    if ratio:
        # Calculate the travel distance ratio relative to the initial goal distance
        travel_distance = torch.max(
            goal_cmd_generator.distance_traveled, goal_cmd_generator.initial_distance_to_goal
        )
        travel_distance_ratio = goal_cmd_generator.initial_distance_to_goal / (travel_distance + 1e-6)
    else:
        travel_distance_ratio = torch.ones_like(reward)

    reward = reward * reward_mask.float() * travel_distance_ratio

    return reward


def backward_movement_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Small penalty for backward movement as a regularization term.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Penalty [0, +1] based on backward velocity (to be used with negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the penalty
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    # Only penalize negative forward velocity (backward movement)
    backward_velocity = torch.clamp(-forward_velocity, min=0.0, max=1.0)
    return backward_velocity


def planar_acceleration_penalty(
    env: ManagerBasedRLEnv,
    action_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize only the planar acceleration command that exceeds the configured threshold."""
    action_term = env.action_manager.get_term(action_name)
    planar_acceleration = getattr(action_term, "preclipped_actions", action_term.processed_actions)[:, :2]
    excess_acceleration = torch.clamp(torch.abs(planar_acceleration) - threshold, min=0.0)
    return torch.mean(torch.clamp(excess_acceleration, max=1.0), dim=1)


def guidance_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    clamp_delta: float = 0.3,
    lateral_sigma: float = 0.6,
) -> torch.Tensor:
    """Reward positive progress along the guidance centerline.

    The reward only considers forward progress and is down-weighted when the robot drifts
    away from the guidance corridor.
    """
    goal_cmd_generator = env.command_manager._terms[command_name]
    current_progress, lateral_error = goal_cmd_generator._project_guidance_state(
        positions_xy=goal_cmd_generator.robot.data.root_pos_w[:, :2],
        guidance_ids=goal_cmd_generator.current_guidance_ids,
    )
    progress_delta = current_progress - goal_cmd_generator.guidance_progress
    positive_delta = torch.clamp(progress_delta, min=0.0, max=clamp_delta)
    corridor_alignment = torch.exp(-torch.square(lateral_error / max(lateral_sigma, 1e-6)))
    return (positive_delta / max(clamp_delta, 1e-6)) * corridor_alignment


def guidance_wrong_way_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    clamp_delta: float = 0.2,
) -> torch.Tensor:
    """Penalize moving backward along the guidance centerline."""
    goal_cmd_generator = env.command_manager._terms[command_name]
    current_progress, _ = goal_cmd_generator._project_guidance_state(
        positions_xy=goal_cmd_generator.robot.data.root_pos_w[:, :2],
        guidance_ids=goal_cmd_generator.current_guidance_ids,
    )
    progress_delta = current_progress - goal_cmd_generator.guidance_progress
    backward_delta = torch.clamp(-progress_delta, min=0.0, max=clamp_delta)
    return backward_delta / max(clamp_delta, 1e-6)


def guidance_lateral_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.6,
) -> torch.Tensor:
    """Penalize lateral deviation from the guidance centerline with smooth saturation."""
    goal_cmd_generator = env.command_manager._terms[command_name]
    _, lateral_error = goal_cmd_generator._project_guidance_state(
        positions_xy=goal_cmd_generator.robot.data.root_pos_w[:, :2],
        guidance_ids=goal_cmd_generator.current_guidance_ids,
    )
    normalized_error = lateral_error / max(sigma, 1e-6)
    return 1.0 - torch.exp(-0.5 * torch.square(normalized_error))
