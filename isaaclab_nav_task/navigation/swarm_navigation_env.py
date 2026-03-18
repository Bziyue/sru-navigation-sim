"""Direct MARL environment for static region-to-region drone swarm navigation."""

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
    """Shared-policy swarm navigation that preserves the original SRU visual architecture."""

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

        self._actions = torch.zeros((self.num_envs, self.swarm_size, 3), dtype=torch.float32, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._root_velocity_command = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self._goal_pos_w = torch.zeros((self.num_envs, self.swarm_size, 3), dtype=torch.float32, device=self.device)
        self._goal_steps = torch.zeros((self.num_envs, self.swarm_size), dtype=torch.long, device=self.device)
        self._required_steps_at_goal = max(1, int(round(self.cfg.required_goal_hold_time_s / self.step_dt)))

        self._guidance_progress = torch.zeros((self.num_envs, self.swarm_size), dtype=torch.float32, device=self.device)
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
                "guidance_progress",
                "guidance_wrong_way",
                "guidance_lateral_error",
                "reach_goal_soft",
                "reach_goal_tight",
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

    def _compute_goal_command(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        goal_in_body = quat_apply_inverse(quat_w.reshape(-1, 4), (self._goal_pos_w - pos_w).reshape(-1, 3)).view(
            self.num_envs, self.swarm_size, 3
        )
        distance = torch.linalg.norm(goal_in_body, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = goal_in_body / distance
        log_distance = torch.log(distance + 1.0)
        return torch.cat([direction, log_distance], dim=-1)

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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        pos_w, quat_w, relative_pos_b, agent_dir = self._compute_pairwise_relative_features()
        lin_vel_b = self._stack_robot_tensor("root_lin_vel_b")
        ang_vel_b = self._stack_robot_tensor("root_ang_vel_b")
        projected_gravity_b = self._stack_robot_tensor("projected_gravity_b")
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
        return torch.linalg.norm(self._goal_pos_w - pos_w, dim=-1)

    def _compute_min_pairwise_distance(self) -> torch.Tensor:
        pos_w = self._stack_robot_tensor("root_pos_w")
        pairwise_dist = torch.cdist(pos_w[:, :, :2], pos_w[:, :, :2])
        pairwise_dist = pairwise_dist.masked_fill(
            torch.eye(self.swarm_size, device=self.device, dtype=torch.bool).unsqueeze(0),
            float("inf"),
        )
        return pairwise_dist.min(dim=-1).values

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        pos_w = self._stack_robot_tensor("root_pos_w")
        flat_positions_xy = pos_w[:, :, :2].reshape(-1, 2)
        flat_guidance_ids = self._shared_guidance_ids.unsqueeze(1).expand(-1, self.swarm_size).reshape(-1)
        progress, lateral_error = self._project_guidance_state(flat_positions_xy, flat_guidance_ids)
        progress = progress.view(self.num_envs, self.swarm_size)
        lateral_error = lateral_error.view(self.num_envs, self.swarm_size)

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

        goal_distance = torch.linalg.norm(self._goal_pos_w - pos_w, dim=-1)
        reach_goal_soft = 1.0 / (1.0 + torch.square(goal_distance / float(self.cfg.soft_goal_radius)))
        reach_goal_tight = 1.0 / (1.0 + torch.square(goal_distance / float(self.cfg.tight_goal_radius)))

        action_rate_l1 = torch.sum(torch.abs(self._actions - self._prev_actions), dim=-1)
        min_other_dist = self._compute_min_pairwise_distance()
        collision_penalty = (min_other_dist < float(self.cfg.agent_collision_distance)).float()
        separation_penalty = torch.clamp(
            (float(self.cfg.agent_separation_distance) - min_other_dist) / float(self.cfg.agent_separation_distance),
            min=0.0,
        )

        rewards = {
            "action_rate_l1": -float(self.cfg.reward_action_rate_l1) * action_rate_l1,
            "guidance_progress": float(self.cfg.reward_guidance_progress) * guidance_progress_reward,
            "guidance_wrong_way": -float(self.cfg.reward_guidance_wrong_way) * guidance_wrong_way_penalty,
            "guidance_lateral_error": -float(self.cfg.reward_guidance_lateral_error) * guidance_lateral_error_penalty,
            "reach_goal_soft": float(self.cfg.reward_reach_goal_soft) * reach_goal_soft,
            "reach_goal_tight": float(self.cfg.reward_reach_goal_tight) * reach_goal_tight,
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
        self._goal_distance.copy_(goal_distance)
        self._prev_actions.copy_(self._actions)

        return {agent: total_reward[:, self._agent_to_index[agent]] for agent in self.agent_ids}

    def _contact_failures_by_agent(self) -> torch.Tensor:
        failures = torch.zeros((self.num_envs, self.swarm_size), dtype=torch.bool, device=self.device)
        for agent in self.agent_ids:
            sensor = self._contact_sensors[agent]
            force_hist = sensor.data.net_forces_w_history
            force_mag = torch.norm(force_hist, dim=-1).amax(dim=1).amax(dim=1)
            failures[:, self._agent_to_index[agent]] = force_mag > float(self.cfg.body_contact_force_threshold)
        return failures

    def _fall_failure(self) -> torch.Tensor:
        min_heights = torch.stack([self._robots[agent].data.root_pos_w[:, 2] for agent in self.agent_ids], dim=1).amin(dim=1)
        return min_heights < float(self.cfg.fall_height_threshold)

    def _collision_failure(self) -> torch.Tensor:
        pos_w = self._stack_robot_tensor("root_pos_w")
        pairwise_dist = torch.cdist(pos_w[:, :, :2], pos_w[:, :, :2])
        upper_dist = pairwise_dist[:, self._upper_tri_rows, self._upper_tri_cols]
        return (upper_dist < float(self.cfg.agent_collision_distance)).any(dim=1)

    def _update_goal_steps(self, goal_distance: torch.Tensor):
        in_goal = goal_distance < float(self.cfg.goal_completion_radius)
        self._goal_steps = torch.where(in_goal, self._goal_steps + 1, torch.zeros_like(self._goal_steps))

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        goal_distance = self._compute_goal_distances()
        self._update_goal_steps(goal_distance)

        self._goal_success_buf = torch.all(self._goal_steps >= self._required_steps_at_goal, dim=1)
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

    def _sample_swarm_points(self, region_ids: torch.Tensor, min_separation: float) -> torch.Tensor:
        points = torch.zeros((len(region_ids), self.swarm_size, 3), dtype=torch.float32, device=self.device)
        for local_index, region_id in enumerate(region_ids.tolist()):
            points[local_index] = self._sample_agent_points(region_id, min_separation=min_separation)
        return points

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

    def _apply_action(self):
        for agent in self.agent_ids:
            index = self._agent_to_index[agent]
            robot = self._robots[agent]
            root_pos = robot.data.root_pos_w.clone()
            root_quat = yaw_quat(robot.data.root_quat_w)
            pose = torch.cat((root_pos, root_quat), dim=-1)
            pose[:, 2] = float(self.cfg.flight_height)
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
        if torch.any(self.episode_length_buf[env_ids] > 0):
            for key, value in self._episode_sums.items():
                shared_log[f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item()
                value[env_ids] = 0.0
        else:
            for value in self._episode_sums.values():
                value[env_ids] = 0.0

        shared_log["Episode_Termination/contact"] = torch.count_nonzero(self._contact_failure_buf[env_ids]).item()
        shared_log["Episode_Termination/agent_collision"] = torch.count_nonzero(self._collision_failure_buf[env_ids]).item()
        shared_log["Episode_Termination/fall"] = torch.count_nonzero(self._fall_failure_buf[env_ids]).item()
        shared_log["Episode_Termination/goal_success"] = torch.count_nonzero(self._goal_success_buf[env_ids]).item()
        shared_log["Episode_Termination/time_out"] = torch.count_nonzero(self._timeout_buf[env_ids]).item()

        super()._reset_idx(env_ids)

        pair_indices = torch.randint(0, len(self.guidance_pair_ids), (len(env_ids),), device=self.device)
        region_pairs = self.guidance_pair_ids[pair_indices]
        spawn_region_ids = region_pairs[:, 0]
        goal_region_ids = region_pairs[:, 1]

        spawn_points = self._sample_swarm_points(spawn_region_ids, min_separation=float(self.cfg.agent_spawn_separation))
        goal_points = self._sample_swarm_points(goal_region_ids, min_separation=float(self.cfg.agent_goal_separation))
        spawn_points[..., 2] = float(self.cfg.flight_height)
        goal_points[..., 2] = float(self.cfg.flight_height)
        self.scene.env_origins[env_ids] = spawn_points.mean(dim=1)

        yaws = (torch.rand((len(env_ids), self.swarm_size), device=self.device) * 2.0 - 1.0) * math.pi
        self._write_robot_state(env_ids, spawn_points, yaws)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        for agent in self.agent_ids:
            self.actions[agent][env_ids] = 0.0

        self._goal_pos_w[env_ids] = goal_points
        self._goal_steps[env_ids] = 0
        self._shared_guidance_ids[env_ids] = pair_indices
        self._hard_failure_buf[env_ids] = False
        self._goal_success_buf[env_ids] = False
        self._timeout_buf[env_ids] = False
        self._contact_failure_buf[env_ids] = False
        self._collision_failure_buf[env_ids] = False
        self._fall_failure_buf[env_ids] = False

        flat_spawn_xy = spawn_points[:, :, :2].reshape(-1, 2)
        flat_guidance_ids = pair_indices.unsqueeze(1).expand(-1, self.swarm_size).reshape(-1)
        progress, lateral_error = self._project_guidance_state(flat_spawn_xy, flat_guidance_ids)
        self._guidance_progress[env_ids] = progress.view(len(env_ids), self.swarm_size)
        self._prev_guidance_progress[env_ids] = self._guidance_progress[env_ids]
        self._guidance_lateral_error[env_ids] = lateral_error.view(len(env_ids), self.swarm_size)
        self._goal_distance[env_ids] = torch.linalg.norm(goal_points - spawn_points, dim=-1)

        for agent in self.agent_ids:
            self.extras[agent] = {"log": shared_log}

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis
