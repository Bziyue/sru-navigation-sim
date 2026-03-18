"""Direct MARL environment for static region-to-region drone swarm cohesion navigation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_nav_task.navigation.mdp.events import setup_static_scan_world
from isaaclab_nav_task.navigation.mdp.navigation.static_region_goal_commands import (
    _build_region_point_sets,
    _load_guidance_centerlines,
)
from isaaclab_nav_task.navigation.mdp.observations import (
    depth_image_noisy_delayed,
    depth_image_prefect,
    height_scan_feat,
)


class _IdentityDepthDelayManager:
    """Minimal delay manager so policy depth can reuse the existing noisy depth helper."""

    def compute_delayed_depth(self, depth: torch.Tensor, camera_name: str) -> torch.Tensor:
        del camera_name
        return depth


class DroneSwarmStaticNavigationEnv(DirectMARLEnv):
    """Shared-policy swarm cohesion task that preserves the original SRU visual architecture."""

    cfg: "DroneSwarmStaticNavigationEnvCfg"

    def __init__(self, cfg: "DroneSwarmStaticNavigationEnvCfg", render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.agent_ids = list(self.cfg.possible_agents)
        self.swarm_size = len(self.agent_ids)
        self._agent_to_index = {agent: index for index, agent in enumerate(self.agent_ids)}

        self._robots: dict[str, Articulation] = {
            agent: self.scene.articulations[f"robot_{index}"] for index, agent in enumerate(self.agent_ids)
        }
        self._contact_sensors: dict[str, ContactSensor] = {
            agent: self.scene.sensors[f"contact_forces_{index}"] for index, agent in enumerate(self.agent_ids)
        }
        self._depth_sensor_cfgs = {
            agent: SceneEntityCfg(f"raycast_camera_{index}") for index, agent in enumerate(self.agent_ids)
        }
        self._height_sensor_cfgs = {
            agent: SceneEntityCfg(f"height_scanner_critic_{index}") for index, agent in enumerate(self.agent_ids)
        }

        self.delay_manager = _IdentityDepthDelayManager()

        self._upper_tri_rows, self._upper_tri_cols = torch.triu_indices(
            self.swarm_size, self.swarm_size, offset=1, device=self.device
        )

        self._actions = torch.zeros((self.num_envs, self.swarm_size, 4), dtype=torch.float32, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._root_velocity_command = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._target_heights = torch.full(
            (self.num_envs, self.swarm_size), float(self.cfg.nominal_height), dtype=torch.float32, device=self.device
        )

        self._goal_center_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._goal_hold_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._required_steps_at_goal = max(1, int(round(self.cfg.required_goal_hold_time_s / self.step_dt)))
        self._curriculum_stage_index = 0
        self._curriculum_success_rate_ema = 0.0
        self._curriculum_completed_episodes = 0
        self._contact_failure_steps = torch.zeros(
            (self.num_envs, self.swarm_size), dtype=torch.long, device=self.device
        )

        self._guidance_progress = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._prev_guidance_progress = torch.zeros_like(self._guidance_progress)
        self._guidance_lateral_error = torch.zeros_like(self._guidance_progress)
        self._goal_distance = torch.zeros_like(self._guidance_progress)
        self._shared_guidance_ids = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self._hard_failure_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._goal_success_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._timeout_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._contact_failure_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._collision_failure_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._fall_failure_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            for key in [
                "action_rate_l1",
                "height_band_violation",
                "height_action_rate_l1",
                "cruise_height_l2",
                "guidance_progress",
                "guidance_wrong_way",
                "guidance_lateral_error",
                "centroid_goal_soft",
                "centroid_goal_tight",
                "cohesion_dispersion_l1",
                "agent_collision",
                "agent_separation",
                "team_success",
                "episode_failure",
            ]
        }

        region_point_sets, _ = _build_region_point_sets(
            surface_bbox_data_path=self.cfg.surface_bbox_data_path,
            map_mesh_prim_path=self.cfg.static_collision_mesh_prim_path,
            flight_height=float(self.cfg.flight_height),
            point_clearance=float(self.cfg.point_clearance),
            grid_spacing=float(self.cfg.safe_point_grid_spacing),
            precomputed_safe_points_path=self.cfg.precomputed_safe_points_path,
        )
        self.region_safe_points = [
            torch.as_tensor(region["safe_points"], dtype=torch.float32, device=self.device) for region in region_point_sets
        ]

        (
            self.guidance_paths_xy,
            self.guidance_arc_lengths,
            self.guidance_path_lengths,
            self.guidance_pair_ids,
        ) = _load_guidance_centerlines(
            guidance_trajectories_data_path=self.cfg.guidance_trajectories_data_path,
            flight_height=float(self.cfg.flight_height),
            eval_dt=float(self.cfg.guidance_trajectory_eval_dt),
            arc_length_spacing=float(self.cfg.guidance_arc_length_spacing),
            num_regions=len(region_point_sets),
        )
        self.guidance_paths_xy = self.guidance_paths_xy.to(self.device)
        self.guidance_arc_lengths = self.guidance_arc_lengths.to(self.device)
        self.guidance_path_lengths = self.guidance_path_lengths.to(self.device)
        self.guidance_pair_ids = self.guidance_pair_ids.to(self.device)

        if bool(self.cfg.curriculum_enabled):
            self._apply_curriculum_stage(stage_index=0)

    def _setup_scene(self):
        setup_static_scan_world(
            self,
            env_ids=None,
            source_prim_expr=self.cfg.static_visual_mesh_prim_path,
            output_prim_path=self.cfg.static_collision_mesh_prim_path,
            walls_root_path="/World/Boundaries",
            wall_padding=float(self.cfg.boundary_wall_padding),
            hide_merged_mesh=True,
        )

    def _stack_robot_tensor(self, attr: str) -> torch.Tensor:
        return torch.stack([getattr(self._robots[agent].data, attr) for agent in self.agent_ids], dim=1)

    def _compute_pairwise_relative_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_w = self._stack_robot_tensor("root_pos_w")
        quat_w = self._stack_robot_tensor("root_quat_w")

        pos_diff_grouped = pos_w.unsqueeze(1) - pos_w.unsqueeze(2)
        non_diag_mask = ~torch.eye(self.swarm_size, dtype=torch.bool, device=self.device)
        relative_pos_w = pos_diff_grouped[:, non_diag_mask].reshape(
            self.num_envs, self.swarm_size, self.swarm_size - 1, 3
        )

        quat_w_expanded = quat_w.unsqueeze(2).expand(-1, -1, self.swarm_size - 1, -1)
        relative_pos_b = quat_apply_inverse(
            quat_w_expanded.reshape(-1, 4), relative_pos_w.reshape(-1, 3)
        ).view(self.num_envs, self.swarm_size, self.swarm_size - 1, 3)
        relative_dist = relative_pos_b.norm(dim=-1).clamp_min(1e-6)
        agent_dir = relative_pos_b / relative_dist.unsqueeze(-1)
        return pos_w, quat_w, relative_pos_b, agent_dir

    def _compute_swarm_centroid(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w.mean(dim=1)

    def _compute_goal_command(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        goal_center_w = self._goal_center_w.unsqueeze(1).expand(-1, self.swarm_size, -1)
        goal_in_body = quat_apply_inverse(quat_w.reshape(-1, 4), (goal_center_w - pos_w).reshape(-1, 3)).view(
            self.num_envs, self.swarm_size, 3
        )
        distance = torch.linalg.norm(goal_in_body, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = goal_in_body / distance
        log_distance = torch.log(distance + 1.0)
        return torch.cat([direction, log_distance], dim=-1)

    def _compute_goal_group_metrics(
        self, pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_center_xy = self._goal_center_w[:, :2]
        centroid_w = self._compute_swarm_centroid(pos_w)
        centroid_goal_distance_xy = torch.linalg.norm(centroid_w[:, :2] - goal_center_xy, dim=-1)
        agent_goal_distance_xy = torch.linalg.norm(pos_w[:, :, :2] - goal_center_xy.unsqueeze(1), dim=-1)
        agent_to_centroid_distance_xy = torch.linalg.norm(pos_w[:, :, :2] - centroid_w[:, :2].unsqueeze(1), dim=-1)
        return centroid_goal_distance_xy, agent_goal_distance_xy, agent_to_centroid_distance_xy, centroid_w

    def _encode_depth_features(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        policy_depth = {}
        critic_depth = {}
        for agent in self.agent_ids:
            policy_depth[agent] = depth_image_noisy_delayed(self, self._depth_sensor_cfgs[agent])
            critic_depth[agent] = depth_image_prefect(self, self._depth_sensor_cfgs[agent])
        return policy_depth, critic_depth

    def _encode_height_features(self) -> dict[str, torch.Tensor]:
        return {
            agent: height_scan_feat(self, self._height_sensor_cfgs[agent])
            for agent in self.agent_ids
        }

    def _get_curriculum_stage_cfg(self, stage_index: int):
        if stage_index == 0:
            return self.cfg.curriculum_stage_1
        if stage_index == 1:
            return self.cfg.curriculum_stage_2
        if stage_index == 2:
            return self.cfg.curriculum_stage_3
        raise ValueError(f"Unsupported curriculum stage index: {stage_index}")

    def _apply_curriculum_stage(self, stage_index: int):
        stage_cfg = self._get_curriculum_stage_cfg(stage_index)
        self._curriculum_stage_index = stage_index

        for attr_name in [
            "body_contact_force_threshold",
            "required_goal_hold_time_s",
            "soft_goal_radius",
            "tight_goal_radius",
            "centroid_goal_completion_radius",
            "agent_goal_completion_radius",
            "cohesion_success_radius",
            "reward_action_rate_l1",
            "reward_height_action_rate_l1",
            "reward_cruise_height_l2",
            "reward_guidance_progress",
            "reward_guidance_wrong_way",
            "reward_guidance_lateral_error",
            "reward_centroid_goal_soft",
            "reward_centroid_goal_tight",
            "reward_cohesion_dispersion_l1",
            "reward_agent_collision",
            "reward_agent_separation",
            "reward_team_success",
            "reward_episode_failure",
        ]:
            setattr(self.cfg, attr_name, getattr(stage_cfg, attr_name))

        self._required_steps_at_goal = max(1, int(round(self.cfg.required_goal_hold_time_s / self.step_dt)))

    def _update_curriculum(self, completed_env_ids: torch.Tensor):
        if not bool(self.cfg.curriculum_enabled) or completed_env_ids.numel() == 0:
            return

        batch_success_rate = self._goal_success_buf[completed_env_ids].float().mean().item()
        alpha = float(self.cfg.curriculum_success_rate_ema_alpha)
        if self._curriculum_completed_episodes == 0:
            self._curriculum_success_rate_ema = batch_success_rate
        else:
            self._curriculum_success_rate_ema = (
                (1.0 - alpha) * self._curriculum_success_rate_ema + alpha * batch_success_rate
            )
        self._curriculum_completed_episodes += int(completed_env_ids.numel())

        thresholds = self.cfg.curriculum_success_rate_thresholds
        min_completed = self.cfg.curriculum_min_completed_episodes
        if (
            self._curriculum_stage_index == 0
            and self._curriculum_completed_episodes >= int(min_completed[0])
            and self._curriculum_success_rate_ema >= float(thresholds[0])
        ):
            self._apply_curriculum_stage(stage_index=1)
        elif (
            self._curriculum_stage_index == 1
            and self._curriculum_completed_episodes >= int(min_completed[1])
            and self._curriculum_success_rate_ema >= float(thresholds[1])
        ):
            self._apply_curriculum_stage(stage_index=2)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        pos_w, quat_w, relative_pos_b, agent_dir = self._compute_pairwise_relative_features()
        lin_vel_b = self._stack_robot_tensor("root_lin_vel_b")
        ang_vel_b = self._stack_robot_tensor("root_ang_vel_b")
        projected_gravity_b = self._stack_robot_tensor("projected_gravity_b")
        base_pos_z = pos_w[:, :, 2:3]
        goal_command = self._compute_goal_command(pos_w, quat_w)
        policy_depth, critic_depth = self._encode_depth_features()
        critic_height = self._encode_height_features()
        time_normalized = self.episode_length_buf.float().unsqueeze(-1) / float(self.max_episode_length)

        relative_dist = relative_pos_b.norm(dim=-1).clamp_min(1e-6)
        policy_intensity = torch.clamp(
            1.0 - relative_dist / float(self.cfg.max_agent_distance_policy), min=0.0
        ).unsqueeze(-1)
        visible_mask = (relative_dist < float(self.cfg.max_agent_distance_policy)).unsqueeze(-1).float()
        policy_others = torch.cat(
            [agent_dir * policy_intensity * visible_mask, policy_intensity, visible_mask],
            dim=-1,
        )

        critic_intensity = torch.clamp(
            1.0 - relative_dist / float(self.cfg.max_agent_distance_critic), min=0.0
        ).unsqueeze(-1)
        critic_visible_mask = (relative_dist < float(self.cfg.max_agent_distance_critic)).unsqueeze(-1).float()
        critic_others = torch.cat(
            [agent_dir * critic_intensity * critic_visible_mask, critic_intensity, critic_visible_mask],
            dim=-1,
        )

        observations: dict[str, dict[str, torch.Tensor]] = {}
        for agent in self.agent_ids:
            index = self._agent_to_index[agent]
            policy_proprio = torch.cat(
                [
                    lin_vel_b[:, index],
                    ang_vel_b[:, index],
                    projected_gravity_b[:, index],
                    base_pos_z[:, index],
                    self._actions[:, index],
                    goal_command[:, index],
                    policy_others[:, index].flatten(start_dim=1),
                ],
                dim=-1,
            )
            critic_proprio = torch.cat(
                [
                    lin_vel_b[:, index],
                    ang_vel_b[:, index],
                    projected_gravity_b[:, index],
                    base_pos_z[:, index],
                    self._actions[:, index],
                    goal_command[:, index],
                    critic_others[:, index].flatten(start_dim=1),
                ],
                dim=-1,
            )
            observations[agent] = {
                "policy": torch.cat([policy_proprio, policy_depth[agent]], dim=-1),
                "critic": torch.cat([critic_proprio, time_normalized, critic_height[agent], critic_depth[agent]], dim=-1),
            }
        return observations

    def _get_states(self) -> None:
        return None

    def _project_guidance_state(self, positions_xy: torch.Tensor, guidance_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        path_xy = self.guidance_paths_xy[guidance_ids]
        arc_lengths = self.guidance_arc_lengths[guidance_ids]
        path_lengths = self.guidance_path_lengths[guidance_ids]

        start_points = path_xy[:, :-1, :]
        end_points = path_xy[:, 1:, :]
        segment_vectors = end_points - start_points
        segment_lengths_sq = torch.sum(segment_vectors * segment_vectors, dim=-1).clamp_min(1e-9)
        segment_lengths = torch.sqrt(segment_lengths_sq)

        rel_positions = positions_xy.unsqueeze(1) - start_points
        raw_t = torch.sum(rel_positions * segment_vectors, dim=-1) / segment_lengths_sq
        clamped_t = torch.clamp(raw_t, min=0.0, max=1.0)
        projected_points = start_points + clamped_t.unsqueeze(-1) * segment_vectors

        deltas = positions_xy.unsqueeze(1) - projected_points
        distances_sq = torch.sum(deltas * deltas, dim=-1)

        max_segments = start_points.shape[1]
        valid_segment_mask = torch.arange(max_segments, device=self.device).unsqueeze(0) < (path_lengths - 1).unsqueeze(1)
        distances_sq = torch.where(valid_segment_mask, distances_sq, torch.full_like(distances_sq, float("inf")))

        closest_segment_indices = torch.argmin(distances_sq, dim=1)
        lateral_error = torch.sqrt(
            torch.gather(distances_sq, 1, closest_segment_indices.unsqueeze(1)).squeeze(1).clamp_min(0.0)
        )
        segment_start_arc = torch.gather(arc_lengths[:, :-1], 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        closest_segment_t = torch.gather(clamped_t, 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        closest_segment_length = torch.gather(segment_lengths, 1, closest_segment_indices.unsqueeze(1)).squeeze(1)
        progress = segment_start_arc + closest_segment_t * closest_segment_length
        return progress, lateral_error

    def _compute_goal_distances(self) -> torch.Tensor:
        pos_w = self._stack_robot_tensor("root_pos_w")
        return torch.linalg.norm(self._goal_center_w.unsqueeze(1) - pos_w, dim=-1)

    def _compute_min_pairwise_distance(self) -> torch.Tensor:
        pos_w = self._stack_robot_tensor("root_pos_w")
        pairwise_dist = torch.cdist(pos_w, pos_w)
        pairwise_dist = pairwise_dist.masked_fill(
            torch.eye(self.swarm_size, device=self.device, dtype=torch.bool).unsqueeze(0),
            float("inf"),
        )
        return pairwise_dist.min(dim=-1).values

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        pos_w = self._stack_robot_tensor("root_pos_w")
        centroid_w = self._compute_swarm_centroid(pos_w)
        progress, lateral_error = self._project_guidance_state(centroid_w[:, :2], self._shared_guidance_ids)
        progress_delta = progress - self._guidance_progress
        positive_delta = torch.clamp(progress_delta, min=0.0, max=float(self.cfg.guidance_progress_clamp))
        backward_delta = torch.clamp(-progress_delta, min=0.0, max=float(self.cfg.guidance_wrong_way_clamp))
        corridor_alignment = torch.exp(
            -torch.square(lateral_error / max(float(self.cfg.guidance_corridor_sigma), 1e-6))
        )
        guidance_progress_reward = (positive_delta / max(float(self.cfg.guidance_progress_clamp), 1e-6)) * corridor_alignment
        guidance_wrong_way_penalty = backward_delta / max(float(self.cfg.guidance_wrong_way_clamp), 1e-6)
        guidance_lateral_error_penalty = 1.0 - torch.exp(
            -0.5 * torch.square(lateral_error / max(float(self.cfg.guidance_lateral_sigma), 1e-6))
        )

        (
            centroid_goal_distance_xy,
            _agent_goal_distance_xy,
            agent_to_centroid_distance_xy,
            _,
        ) = self._compute_goal_group_metrics(pos_w)
        centroid_goal_soft = 1.0 / (
            1.0 + torch.square(centroid_goal_distance_xy / float(self.cfg.soft_goal_radius))
        )
        centroid_goal_tight = 1.0 / (
            1.0 + torch.square(centroid_goal_distance_xy / float(self.cfg.tight_goal_radius))
        )
        cohesion_dispersion_l1 = torch.clamp(
            agent_to_centroid_distance_xy - float(self.cfg.cohesion_soft_radius),
            min=0.0,
            max=0.35,
        ).mean(dim=1)

        action_rate_l1 = torch.sum(torch.abs(self._actions - self._prev_actions), dim=-1)
        height = pos_w[:, :, 2]
        below = torch.clamp(float(self.cfg.min_height) - height, min=0.0)
        above = torch.clamp(height - float(self.cfg.max_height), min=0.0)
        height_band_violation = torch.square(below + above)
        height_action_rate_l1 = torch.abs(self._actions[:, :, 3] - self._prev_actions[:, :, 3])
        cruise_height_l2 = torch.square(height - float(self.cfg.nominal_height)) * (
            centroid_goal_distance_xy.unsqueeze(1) > float(self.cfg.cruise_height_release_distance)
        ).float()
        min_other_dist = self._compute_min_pairwise_distance()
        collision_penalty = (min_other_dist < float(self.cfg.agent_collision_distance)).float()
        separation_penalty = torch.clamp(
            (float(self.cfg.agent_separation_distance) - min_other_dist) / float(self.cfg.agent_separation_distance),
            min=0.0,
        )

        rewards = {
            "action_rate_l1": -float(self.cfg.reward_action_rate_l1) * action_rate_l1,
            "height_band_violation": -float(self.cfg.reward_height_band_violation) * height_band_violation,
            "height_action_rate_l1": -float(self.cfg.reward_height_action_rate_l1) * height_action_rate_l1,
            "cruise_height_l2": -float(self.cfg.reward_cruise_height_l2) * cruise_height_l2,
            "guidance_progress": float(self.cfg.reward_guidance_progress)
            * guidance_progress_reward.unsqueeze(1).expand(-1, self.swarm_size),
            "guidance_wrong_way": -float(self.cfg.reward_guidance_wrong_way)
            * guidance_wrong_way_penalty.unsqueeze(1).expand(-1, self.swarm_size),
            "guidance_lateral_error": -float(self.cfg.reward_guidance_lateral_error)
            * guidance_lateral_error_penalty.unsqueeze(1).expand(-1, self.swarm_size),
            "centroid_goal_soft": float(self.cfg.reward_centroid_goal_soft)
            * centroid_goal_soft.unsqueeze(1).expand(-1, self.swarm_size),
            "centroid_goal_tight": float(self.cfg.reward_centroid_goal_tight)
            * centroid_goal_tight.unsqueeze(1).expand(-1, self.swarm_size),
            "cohesion_dispersion_l1": -float(self.cfg.reward_cohesion_dispersion_l1)
            * cohesion_dispersion_l1.unsqueeze(1).expand(-1, self.swarm_size),
            "agent_collision": -float(self.cfg.reward_agent_collision) * collision_penalty,
            "agent_separation": -float(self.cfg.reward_agent_separation) * separation_penalty,
            "team_success": float(self.cfg.reward_team_success) * self._goal_success_buf.float().unsqueeze(1),
            "episode_failure": -float(self.cfg.reward_episode_failure) * self._hard_failure_buf.float().unsqueeze(1),
        }

        total_reward = torch.zeros((self.num_envs, self.swarm_size), dtype=torch.float32, device=self.device)
        for key, value in rewards.items():
            total_reward += value
            self._episode_sums[key] += value.mean(dim=1)

        self._prev_guidance_progress.copy_(self._guidance_progress)
        self._guidance_progress.copy_(progress)
        self._guidance_lateral_error.copy_(lateral_error)
        self._goal_distance.copy_(centroid_goal_distance_xy)
        self._prev_actions.copy_(self._actions)

        return {agent: total_reward[:, self._agent_to_index[agent]] for agent in self.agent_ids}

    def _contact_failures_by_agent(self) -> torch.Tensor:
        failures = torch.zeros((self.num_envs, self.swarm_size), dtype=torch.bool, device=self.device)
        debounce_steps = max(1, int(self.cfg.contact_failure_debounce_steps))
        for agent in self.agent_ids:
            sensor = self._contact_sensors[agent]
            force_hist = sensor.data.net_forces_w_history
            force_mag = torch.norm(force_hist, dim=-1).amax(dim=1).amax(dim=1)
            agent_index = self._agent_to_index[agent]
            over_threshold = force_mag > float(self.cfg.body_contact_force_threshold)
            self._contact_failure_steps[:, agent_index] = torch.where(
                over_threshold,
                self._contact_failure_steps[:, agent_index] + 1,
                torch.zeros_like(self._contact_failure_steps[:, agent_index]),
            )
            failures[:, agent_index] = self._contact_failure_steps[:, agent_index] >= debounce_steps
        return failures

    def _fall_failure(self) -> torch.Tensor:
        min_heights = torch.stack([self._robots[agent].data.root_pos_w[:, 2] for agent in self.agent_ids], dim=1).amin(dim=1)
        return min_heights < float(self.cfg.fall_height_threshold)

    def _collision_failure(self) -> torch.Tensor:
        pos_w = self._stack_robot_tensor("root_pos_w")
        pairwise_dist = torch.cdist(pos_w, pos_w)
        upper_dist = pairwise_dist[:, self._upper_tri_rows, self._upper_tri_cols]
        return (upper_dist < float(self.cfg.agent_collision_distance)).any(dim=1)

    def _update_goal_steps(self, pos_w: torch.Tensor):
        centroid_goal_distance_xy, agent_goal_distance_xy, agent_to_centroid_distance_xy, _ = self._compute_goal_group_metrics(
            pos_w
        )
        centroid_in_band = centroid_goal_distance_xy <= float(self.cfg.centroid_goal_completion_radius)
        agents_near_goal = agent_goal_distance_xy.max(dim=1).values <= float(self.cfg.agent_goal_completion_radius)
        cohesion_ok = agent_to_centroid_distance_xy.max(dim=1).values <= float(self.cfg.cohesion_success_radius)
        goal_ready = centroid_in_band & agents_near_goal & cohesion_ok
        self._goal_hold_steps = torch.where(goal_ready, self._goal_hold_steps + 1, torch.zeros_like(self._goal_hold_steps))

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pos_w = self._stack_robot_tensor("root_pos_w")
        self._update_goal_steps(pos_w)

        self._goal_success_buf = self._goal_hold_steps >= self._required_steps_at_goal
        self._contact_failure_buf = self._contact_failures_by_agent().any(dim=1)
        self._collision_failure_buf = self._collision_failure()
        self._fall_failure_buf = self._fall_failure()
        self._hard_failure_buf = self._contact_failure_buf | self._collision_failure_buf | self._fall_failure_buf
        self._timeout_buf = self.episode_length_buf >= self.max_episode_length - 1

        terminated = self._hard_failure_buf | self._goal_success_buf
        time_out = self._timeout_buf
        return (
            {agent: terminated.clone() for agent in self.agent_ids},
            {agent: time_out.clone() for agent in self.agent_ids},
        )

    def _sample_agent_points(self, region_id: int, min_separation: float) -> torch.Tensor:
        region_points = self.region_safe_points[region_id]
        if len(region_points) == 0:
            raise RuntimeError(f"Region {region_id} does not contain any safe points.")

        perm = torch.randperm(len(region_points), device=self.device)
        selected: list[torch.Tensor] = []
        for point in region_points[perm]:
            if not selected:
                selected.append(point)
            else:
                prev = torch.stack(selected, dim=0)
                if torch.all(torch.linalg.norm(prev[:, :2] - point[:2], dim=1) >= min_separation):
                    selected.append(point)
            if len(selected) == self.swarm_size:
                break

        while len(selected) < self.swarm_size:
            selected.append(region_points[torch.randint(0, len(region_points), (1,), device=self.device).item()])

        return torch.stack(selected, dim=0)

    def _sample_formation_layout(
        self, region_id: int, formation_radius: float, min_separation: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region_points = self.region_safe_points[region_id]
        if len(region_points) == 0:
            raise RuntimeError(f"Region {region_id} does not contain any safe points.")

        agent_angles = torch.arange(self.swarm_size, device=self.device, dtype=torch.float32) * (2.0 * math.pi / self.swarm_size)
        for _ in range(int(self.cfg.formation_sampling_attempts)):
            center = region_points[torch.randint(0, len(region_points), (1,), device=self.device).item()]
            base_angle = torch.rand((), device=self.device) * (2.0 * math.pi)
            desired_angles = agent_angles + base_angle
            desired_xy = center[:2].unsqueeze(0) + formation_radius * torch.stack(
                [torch.cos(desired_angles), torch.sin(desired_angles)], dim=-1
            )
            desired_z = torch.full((self.swarm_size, 1), center[2], dtype=torch.float32, device=self.device)
            desired_points = torch.cat([desired_xy, desired_z], dim=-1)

            available_mask = torch.ones(len(region_points), dtype=torch.bool, device=self.device)
            selected_indices: list[torch.Tensor] = []
            success = True
            for desired_point in desired_points:
                candidate_indices = torch.nonzero(available_mask, as_tuple=False).squeeze(-1)
                candidates = region_points[candidate_indices]
                xy_error = torch.linalg.norm(candidates[:, :2] - desired_point[:2], dim=1)
                z_error = torch.abs(candidates[:, 2] - desired_point[2])
                score = xy_error + float(self.cfg.formation_z_score_weight) * z_error
                best_local = torch.argmin(score)
                if xy_error[best_local] > float(self.cfg.formation_assignment_max_error):
                    success = False
                    break
                best_global = candidate_indices[best_local]
                selected_indices.append(best_global)
                available_mask[best_global] = False

            if success:
                points = region_points[torch.stack(selected_indices)]
                pairwise_xy = torch.cdist(points[:, :2].unsqueeze(0), points[:, :2].unsqueeze(0)).squeeze(0)
                pairwise_xy.fill_diagonal_(float("inf"))
                if pairwise_xy.min() >= min_separation:
                    return center, points

        fallback_points = self._sample_agent_points(region_id, min_separation=min_separation)
        return fallback_points.mean(dim=0), fallback_points

    def _sample_swarm_layouts(
        self, region_ids: torch.Tensor, formation_radius: float, min_separation: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = torch.zeros((len(region_ids), 3), dtype=torch.float32, device=self.device)
        points = torch.zeros((len(region_ids), self.swarm_size, 3), dtype=torch.float32, device=self.device)
        for local_index, region_id in enumerate(region_ids.tolist()):
            centers[local_index], points[local_index] = self._sample_formation_layout(
                region_id,
                formation_radius=formation_radius,
                min_separation=min_separation,
            )
        return centers, points

    def _sample_region_centers(self, region_ids: torch.Tensor) -> torch.Tensor:
        centers = torch.zeros((len(region_ids), 3), dtype=torch.float32, device=self.device)
        for local_index, region_id in enumerate(region_ids.tolist()):
            region_points = self.region_safe_points[region_id]
            centers[local_index] = region_points[torch.randint(0, len(region_points), (1,), device=self.device).item()]
        return centers

    def _write_robot_state(self, env_ids: torch.Tensor, positions: torch.Tensor, yaws: torch.Tensor):
        flat_yaws = yaws.reshape(-1)
        zeros = torch.zeros_like(flat_yaws)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, flat_yaws).view(len(env_ids), self.swarm_size, 4)
        zero_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)

        for agent in self.agent_ids:
            index = self._agent_to_index[agent]
            robot = self._robots[agent]
            robot.reset(env_ids)
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            root_pose = torch.cat((positions[:, index], quats[:, index]), dim=-1)
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        for agent in self.agent_ids:
            index = self._agent_to_index[agent]
            clamped = actions[agent].clamp(-1.0, 1.0)
            self.actions[agent] = clamped
            self._actions[:, index].copy_(clamped)
            current_heights = self._robots[agent].data.root_pos_w[:, 2]
            self._target_heights[:, index] = torch.clamp(
                current_heights + clamped[:, 3] * float(self.cfg.action_scale_z),
                min=float(self.cfg.min_height),
                max=float(self.cfg.max_height),
            )

    def _apply_action(self):
        for agent in self.agent_ids:
            index = self._agent_to_index[agent]
            robot = self._robots[agent]
            root_pos = robot.data.root_pos_w.clone()
            root_quat = yaw_quat(robot.data.root_quat_w)
            pose = torch.cat((root_pos, root_quat), dim=-1)
            pose[:, 2] = self._target_heights[:, index]
            robot.write_root_pose_to_sim(pose)

            yaw = math_utils.euler_xyz_from_quat(root_quat)[2]
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            vx_body = self._actions[:, index, 0] * float(self.cfg.action_scale_xy)
            vy_body = self._actions[:, index, 1] * float(self.cfg.action_scale_xy)

            self._root_velocity_command.zero_()
            self._root_velocity_command[:, 0] = cos_yaw * vx_body - sin_yaw * vy_body
            self._root_velocity_command[:, 1] = sin_yaw * vx_body + cos_yaw * vy_body
            self._root_velocity_command[:, 5] = self._actions[:, index, 2] * float(self.cfg.action_scale_yaw)
            robot.write_root_velocity_to_sim(self._root_velocity_command)

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        shared_log = {}
        completed_env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if completed_env_ids.numel() > 0:
            self._update_curriculum(completed_env_ids)
            for key, value in self._episode_sums.items():
                shared_log[f"Episode_Reward/{key}"] = torch.mean(value[completed_env_ids]).item()
        else:
            for key in self._episode_sums:
                shared_log[f"Episode_Reward/{key}"] = 0.0

        for value in self._episode_sums.values():
            value[env_ids] = 0.0

        stats_env_ids = completed_env_ids if completed_env_ids.numel() > 0 else env_ids
        shared_log["Episode_Termination/contact"] = torch.count_nonzero(self._contact_failure_buf[stats_env_ids]).item()
        shared_log["Episode_Termination/agent_collision"] = torch.count_nonzero(
            self._collision_failure_buf[stats_env_ids]
        ).item()
        shared_log["Episode_Termination/fall"] = torch.count_nonzero(self._fall_failure_buf[stats_env_ids]).item()
        shared_log["Episode_Termination/goal_success"] = torch.count_nonzero(self._goal_success_buf[stats_env_ids]).item()
        shared_log["Episode_Termination/time_out"] = torch.count_nonzero(self._timeout_buf[stats_env_ids]).item()
        shared_log["Curriculum/stage"] = float(self._curriculum_stage_index + 1)
        shared_log["Curriculum/success_rate_ema"] = float(self._curriculum_success_rate_ema)
        shared_log["Curriculum/completed_episodes"] = float(self._curriculum_completed_episodes)

        super()._reset_idx(env_ids)

        pair_indices = torch.randint(0, len(self.guidance_pair_ids), (len(env_ids),), device=self.device)
        region_pairs = self.guidance_pair_ids[pair_indices]
        spawn_region_ids = region_pairs[:, 0]
        goal_region_ids = region_pairs[:, 1]

        spawn_centers, spawn_points = self._sample_swarm_layouts(
            spawn_region_ids,
            formation_radius=float(self.cfg.spawn_formation_radius),
            min_separation=float(self.cfg.agent_spawn_separation),
        )
        goal_centers = self._sample_region_centers(goal_region_ids)
        self.scene.env_origins[env_ids] = spawn_centers

        yaws = (torch.rand((len(env_ids), self.swarm_size), device=self.device) * 2.0 - 1.0) * math.pi
        self._write_robot_state(env_ids, spawn_points, yaws)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._target_heights[env_ids] = spawn_points[:, :, 2]
        for agent in self.agent_ids:
            self.actions[agent][env_ids] = 0.0

        self._goal_center_w[env_ids] = goal_centers
        self._goal_hold_steps[env_ids] = 0
        self._shared_guidance_ids[env_ids] = pair_indices
        self._hard_failure_buf[env_ids] = False
        self._goal_success_buf[env_ids] = False
        self._timeout_buf[env_ids] = False
        self._contact_failure_steps[env_ids] = 0
        self._contact_failure_buf[env_ids] = False
        self._collision_failure_buf[env_ids] = False
        self._fall_failure_buf[env_ids] = False

        progress, lateral_error = self._project_guidance_state(spawn_points.mean(dim=1)[:, :2], pair_indices)
        self._guidance_progress[env_ids] = progress
        self._prev_guidance_progress[env_ids] = self._guidance_progress[env_ids]
        self._guidance_lateral_error[env_ids] = lateral_error
        self._goal_distance[env_ids] = torch.linalg.norm(goal_centers[:, :2] - spawn_points.mean(dim=1)[:, :2], dim=-1)

        for agent in self.agent_ids:
            self.extras[agent] = {"log": shared_log}

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis
