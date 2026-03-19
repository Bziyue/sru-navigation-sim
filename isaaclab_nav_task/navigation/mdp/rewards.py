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


def height_band_violation(
    env: "ManagerBasedRLEnv",
    min_height: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize flying below or above the allowed height band."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    below = torch.clamp(min_height - height, min=0.0)
    above = torch.clamp(height - max_height, min=0.0)
    violation = below + above
    return torch.square(violation)


def height_hold_action_l1(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "velocity_command",
    action_index: int = 3,
) -> torch.Tensor:
    """Penalize non-zero height increments so the policy prefers holding altitude."""
    action_term = env.action_manager.get_term(action_term_name)
    return torch.abs(action_term.processed_actions[:, action_index])


def height_action_rate_l1(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "velocity_command",
    action_index: int = 3,
) -> torch.Tensor:
    """Penalize abrupt changes in the height command."""
    action_term = env.action_manager.get_term(action_term_name)
    return torch.abs(action_term.processed_actions[:, action_index] - action_term._prev_processed_actions[:, action_index])


def cruise_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    release_distance: float,
    command_name: str = "robot_goal",
    flat: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from a cruise height except when the robot is close to the goal.

    This keeps the drone near the nominal cruise height during transit, while allowing it
    to match a goal with a different z-value once it gets sufficiently close to the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[command_name]

    if flat:
        goal_distance = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.goal_position_world[:, :2], dim=1)
    else:
        goal_distance = torch.norm(asset.data.root_pos_w[:, :3] - goal_cmd_generator.goal_position_world[:, :3], dim=1)

    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    cruise_mask = goal_distance > release_distance
    return height_error * cruise_mask.float()


def adaptive_cruise_height_l2(
    env: "ManagerBasedRLEnv",
    cruise_height: float,
    xy_follow_start_distance: float,
    xy_follow_full_distance: float,
    height_deadband: float,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize height error against a target that blends from cruise height to goal height.

    Far from the goal, the drone is encouraged to stay near ``cruise_height``. As it approaches
    the goal in the XY plane, the target height smoothly transitions to the goal's z-value so the
    policy can learn the final vertical alignment needed for success.

    A small deadband keeps this term permissive enough for temporary obstacle avoidance maneuvers.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator = env.command_manager._terms[command_name]

    distance_xy = torch.norm(
        asset.data.root_pos_w[:, :2] - goal_cmd_generator.goal_position_world[:, :2],
        dim=1,
    )
    blend_denom = max(float(xy_follow_start_distance - xy_follow_full_distance), 1e-6)
    goal_height_blend = torch.clamp(
        (float(xy_follow_start_distance) - distance_xy) / blend_denom,
        min=0.0,
        max=1.0,
    )

    target_height = (1.0 - goal_height_blend) * float(cruise_height) + goal_height_blend * goal_cmd_generator.goal_position_world[:, 2]
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
    if height_deadband > 0.0:
        height_error = torch.clamp(height_error - float(height_deadband), min=0.0)
    return torch.square(height_error)


def near_goal_z_align_l2(
    env: "ManagerBasedRLEnv",
    xy_activation_distance: float,
    z_deadband: float,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute z error more strongly once the robot is near the goal in XY."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator = env.command_manager._terms[command_name]

    position_error = goal_cmd_generator.goal_position_world[:, :3] - asset.data.root_pos_w[:, :3]
    distance_xy = torch.norm(position_error[:, :2], dim=1)
    goal_z_error_abs = torch.abs(position_error[:, 2])
    z_error = torch.clamp(goal_z_error_abs - float(z_deadband), min=0.0)
    near_goal_mask = (distance_xy < float(xy_activation_distance)).float()
    return torch.square(z_error) * near_goal_mask


def first_goal_reach_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "robot_goal",
    xy_threshold: float | None = None,
    z_threshold: float | None = None,
) -> torch.Tensor:
    """Emit a one-step bonus when the robot first enters the relaxed success gate."""
    goal_cmd_generator = env.command_manager._terms[command_name]
    reach_xy_threshold = float(xy_threshold) if xy_threshold is not None else float(goal_cmd_generator.first_reach_xy_threshold)
    reach_z_threshold = float(z_threshold) if z_threshold is not None else float(goal_cmd_generator.first_reach_z_threshold)
    _, _, in_first_reach_gate = goal_cmd_generator._compute_goal_gate(reach_xy_threshold, reach_z_threshold)
    goal_cmd_generator._update_first_reach_state(in_first_reach_gate)
    bonus = goal_cmd_generator.first_reach_bonus_pending.float()
    goal_cmd_generator.first_reach_bonus_pending[:] = False
    return bonus


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
    gate_xy_threshold: float | None = None,
    gate_z_threshold: float | None = None,
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

    if flat:
        current_gate_mask = xyz_error < sigmoid
    else:
        resolved_gate_xy_threshold = gate_xy_threshold
        resolved_gate_z_threshold = gate_z_threshold
        if resolved_gate_xy_threshold is None:
            resolved_gate_xy_threshold = getattr(goal_cmd_generator, "first_reach_xy_threshold", None)
        if resolved_gate_z_threshold is None:
            resolved_gate_z_threshold = getattr(goal_cmd_generator, "first_reach_z_threshold", None)

        if resolved_gate_xy_threshold is not None and resolved_gate_z_threshold is not None:
            _, _, current_gate_mask = goal_cmd_generator._compute_goal_gate(
                float(resolved_gate_xy_threshold),
                float(resolved_gate_z_threshold),
            )
        else:
            current_gate_mask = xyz_error < sigmoid

    reward_mask = torch.logical_or(timeup_mask, current_gate_mask)

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


def goal_hold_time_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "robot_goal",
    max_hold_time_s: float = 4.0,
) -> torch.Tensor:
    """Reward continuous residence in the goal region with linear saturation."""
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[command_name]
    dwell_time_s = goal_cmd_generator.time_at_goal
    return torch.clamp(dwell_time_s / max(max_hold_time_s, 1e-6), min=0.0, max=1.0)


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


def _guidance_active_mask(goal_cmd_generator, fade_steps: float) -> torch.Tensor:
    """Linearly fade guidance shaping after entering the target region."""
    region_steps = goal_cmd_generator.steps_inside_goal_region
    fade = 1.0 - region_steps / max(float(fade_steps), 1.0)
    return torch.clamp(fade, min=0.0, max=1.0)


def guidance_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    clamp_delta: float = 0.3,
    lateral_sigma: float = 0.6,
    goal_region_fade_steps: float = 15.0,
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
    return (positive_delta / max(clamp_delta, 1e-6)) * corridor_alignment * _guidance_active_mask(
        goal_cmd_generator, goal_region_fade_steps
    )


def guidance_wrong_way_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    clamp_delta: float = 0.2,
    goal_region_fade_steps: float = 15.0,
) -> torch.Tensor:
    """Penalize moving backward along the guidance centerline."""
    goal_cmd_generator = env.command_manager._terms[command_name]
    current_progress, _ = goal_cmd_generator._project_guidance_state(
        positions_xy=goal_cmd_generator.robot.data.root_pos_w[:, :2],
        guidance_ids=goal_cmd_generator.current_guidance_ids,
    )
    progress_delta = current_progress - goal_cmd_generator.guidance_progress
    backward_delta = torch.clamp(-progress_delta, min=0.0, max=clamp_delta)
    return (backward_delta / max(clamp_delta, 1e-6)) * _guidance_active_mask(goal_cmd_generator, goal_region_fade_steps)


def guidance_lateral_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.6,
    goal_region_fade_steps: float = 15.0,
) -> torch.Tensor:
    """Penalize lateral deviation from the guidance centerline with smooth saturation."""
    goal_cmd_generator = env.command_manager._terms[command_name]
    _, lateral_error = goal_cmd_generator._project_guidance_state(
        positions_xy=goal_cmd_generator.robot.data.root_pos_w[:, :2],
        guidance_ids=goal_cmd_generator.current_guidance_ids,
    )
    normalized_error = lateral_error / max(sigma, 1e-6)
    penalty = 1.0 - torch.exp(-0.5 * torch.square(normalized_error))
    return penalty * _guidance_active_mask(goal_cmd_generator, goal_region_fade_steps)
