"""Direct multi-agent drone swarm navigation environment on a static scanned mesh."""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat

from isaaclab_nav_task.navigation.mdp.events import setup_static_scan_world
from isaaclab_nav_task.navigation.mdp.navigation.static_region_goal_commands import (
    _build_region_point_sets,
    _load_guidance_centerlines,
)
from isaaclab_nav_task.navigation.mdp.observations import depth_image_prefect, height_scan_feat

from .config.drone_swarm.swarm_env_cfg import DroneSwarmNavigationEnvCfg


class DroneSwarmNavigationEnv(DirectMARLEnv):
    """Five-drone swarm navigation with shared goal and teammate observations."""

    cfg: DroneSwarmNavigationEnvCfg

    def __init__(self, cfg: DroneSwarmNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.agent_ids = list(self.cfg.possible_agents)
        self.num_agents_cfg = len(self.agent_ids)

        self.robots: dict[str, Articulation] = {
            agent: self.scene.articulations[f"robot_{idx}"] for idx, agent in enumerate(self.agent_ids)
        }
        self.cameras: dict[str, RayCasterCamera] = {
            agent: self.scene.sensors[f"raycast_camera_{idx}"] for idx, agent in enumerate(self.agent_ids)
        }
        self.height_scanners: dict[str, RayCaster] = {
            agent: self.scene.sensors[f"height_scanner_critic_{idx}"] for idx, agent in enumerate(self.agent_ids)
        }
        self.contact_sensors: dict[str, ContactSensor] = {
            agent: self.scene.sensors[f"contact_forces_{idx}"] for idx, agent in enumerate(self.agent_ids)
        }
        self.contact_body_ids = {
            agent: self.contact_sensors[agent].find_bodies("body")[0] for agent in self.agent_ids
        }

        region_point_sets, _ = _build_region_point_sets(
            surface_bbox_data_path=self.cfg.surface_bbox_data_path,
            map_mesh_prim_path=self.cfg.map_mesh_prim_path,
            flight_height=float(self.cfg.flight_height),
            point_clearance=float(self.cfg.point_clearance),
            grid_spacing=float(self.cfg.safe_point_grid_spacing),
            precomputed_safe_points_path=self.cfg.precomputed_safe_points_path,
        )
        self.region_safe_points = [
            torch.as_tensor(region["safe_points"], dtype=torch.float32, device=self.device) for region in region_point_sets
        ]
        (
            guidance_paths_xy,
            guidance_arc_lengths,
            guidance_path_lengths,
            guidance_pair_ids,
        ) = _load_guidance_centerlines(
            guidance_trajectories_data_path=self.cfg.guidance_trajectories_data_path,
            flight_height=self.cfg.flight_height,
            eval_dt=self.cfg.guidance_trajectory_eval_dt,
            arc_length_spacing=self.cfg.guidance_arc_length_spacing,
            num_regions=len(region_point_sets),
        )
        self.guidance_paths_xy = guidance_paths_xy.to(self.device)
        self.guidance_arc_lengths = guidance_arc_lengths.to(self.device)
        self.guidance_path_lengths = guidance_path_lengths.to(self.device)
        self.guidance_pair_ids = guidance_pair_ids.to(self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, dtype=torch.float32, device=self.device)
        self._formation_offsets = torch.tensor(
            self.cfg.initial_formation_offsets_xy, dtype=torch.float32, device=self.device
        )
        self.region_fallback_center_indices = self._build_region_fallback_center_indices()

        self._processed_actions = {
            agent: torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self._previous_actions = {
            agent: torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self._desired_velocity_w = {
            agent: torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }

        self.cluster_spawn_center = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.cluster_goal_center = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.cluster_centroid = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.source_region_ids = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_region_ids = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.current_guidance_ids = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cluster_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_collision = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_fall = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_goal_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.cluster_max_centroid_radius = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.cluster_max_pairwise_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.cluster_target_region_reached = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_target_region_bonus_awarded = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.agent_guidance_progress = {
            agent: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self.agent_prev_guidance_progress = {
            agent: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self.agent_guidance_lateral_error = {
            agent: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self.agent_goal_distance = {
            agent: torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) for agent in self.agent_ids
        }
        self.agent_contact_penalty_active = {
            agent: torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) for agent in self.agent_ids
        }
        self.agent_contact_termination_streak = {
            agent: torch.zeros((self.num_envs,), dtype=torch.long, device=self.device) for agent in self.agent_ids
        }
        self._pending_log: dict[str, torch.Tensor] | None = None

    def _setup_scene(self):
        setup_static_scan_world(
            self,
            None,
            source_prim_expr=self.cfg.static_visual_mesh_prim_path,
            output_prim_path=self.cfg.map_mesh_prim_path,
            walls_root_path=self.cfg.boundary_walls_root_path,
            wall_padding=self.cfg.boundary_wall_padding,
            hide_merged_mesh=True,
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        for agent in self.agent_ids:
            self.actions[agent] = actions[agent].clone()
            self._processed_actions[agent] = torch.tanh(self.actions[agent]) * self._action_scale

    def _apply_action(self):
        for agent, robot in self.robots.items():
            processed = self._processed_actions[agent]
            yaw_only_quat = yaw_quat(robot.data.root_quat_w)
            planar_acc_body = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            planar_acc_body[:, :2] = processed[:, :2]
            planar_acc_world = quat_apply(yaw_only_quat, planar_acc_body)

            desired_vel = self._desired_velocity_w[agent]
            desired_vel[:, :2] += planar_acc_world[:, :2] * self.physics_dt
            desired_vel[:, 2] = 0.0
            speed_xy = torch.linalg.norm(desired_vel[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
            desired_vel[:, :2] = torch.where(
                speed_xy > self.cfg.max_speed,
                desired_vel[:, :2] * (self.cfg.max_speed / speed_xy),
                desired_vel[:, :2],
            )

            root_pose = torch.cat((robot.data.root_pos_w.clone(), yaw_only_quat), dim=-1)
            root_pose[:, 2] = self.cfg.flight_height
            root_velocity = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
            root_velocity[:, :2] = desired_vel[:, :2]
            root_velocity[:, 5] = processed[:, 2]
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        self._update_common_buffers()
        observations: dict[str, dict[str, torch.Tensor]] = {}
        for agent_idx, agent in enumerate(self.agent_ids):
            robot = self.robots[agent]
            goal_delta_w = self.cluster_goal_center - robot.data.root_pos_w
            goal_delta_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_delta_w)
            goal_distance = torch.linalg.norm(goal_delta_b[:, :2], dim=1, keepdim=True)
            goal_command = torch.cat(
                (
                    goal_delta_b[:, :3] / torch.clamp(torch.linalg.norm(goal_delta_b[:, :3], dim=1, keepdim=True), min=1e-6),
                    torch.log1p(goal_distance),
                ),
                dim=1,
            )

            teammate_features = []
            teammate_sort_keys = []
            own_pos = robot.data.root_pos_w
            own_vel = robot.data.root_lin_vel_w
            own_yaw_quat = yaw_quat(robot.data.root_quat_w)
            for other_idx, other_agent in enumerate(self.agent_ids):
                if other_idx == agent_idx:
                    continue
                other_robot = self.robots[other_agent]
                rel_pos_w = other_robot.data.root_pos_w - own_pos
                rel_vel_w = other_robot.data.root_lin_vel_w - own_vel
                rel_pos_b = quat_apply_inverse(own_yaw_quat, rel_pos_w)[:, :2]
                rel_vel_b = quat_apply_inverse(own_yaw_quat, rel_vel_w)[:, :2]
                rel_dist = torch.linalg.norm(rel_pos_w[:, :2], dim=1, keepdim=True)
                visible = (rel_dist <= self.cfg.teammate_observation_radius).float()
                teammate_features.append(
                    torch.cat((rel_pos_b * visible, rel_vel_b * visible, rel_dist * visible, visible), dim=1)
                )
                teammate_sort_keys.append(rel_dist.squeeze(1))

            teammate_feature_stack = torch.stack(teammate_features, dim=1)
            teammate_sort_key_stack = torch.stack(teammate_sort_keys, dim=1)
            teammate_sort_indices = torch.argsort(teammate_sort_key_stack, dim=1)
            teammate_feature_stack = torch.gather(
                teammate_feature_stack,
                1,
                teammate_sort_indices.unsqueeze(-1).expand(-1, -1, teammate_feature_stack.shape[-1]),
            )
            teammate_obs = teammate_feature_stack.reshape(self.num_envs, -1)

            proprio = torch.cat(
                (
                    robot.data.root_lin_vel_b,
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    self.actions[agent],
                    goal_command,
                    teammate_obs,
                ),
                dim=1,
            )

            critic_proprio = torch.cat(
                (
                    proprio,
                    (self.episode_length_buf.unsqueeze(-1).float() / float(self.max_episode_length)),
                ),
                dim=1,
            )
            depth_features = self._get_depth_features(agent)
            height_features = self._get_height_features(agent)

            observations[agent] = {
                "policy": torch.cat((proprio, depth_features), dim=1),
                "critic": torch.cat((critic_proprio, height_features, depth_features), dim=1),
            }

        for agent in self.agent_ids:
            self._previous_actions[agent].copy_(self.actions[agent])
            if self._pending_log is not None:
                self.extras[agent]["log"] = self._pending_log
        self._pending_log = None
        return observations

    def _get_states(self) -> torch.Tensor:
        return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards: dict[str, torch.Tensor] = {}
        pairwise_distances = self._compute_pairwise_distances_xy()
        nearest_distances = self._compute_nearest_neighbor_distances(pairwise_distances)
        centroid_spread = self._compute_centroid_spread()
        max_pairwise_distance = pairwise_distances.amax(dim=(1, 2))
        enter_target_region_bonus = self.cluster_target_region_reached & ~self.cluster_target_region_bonus_awarded
        for agent_idx, agent in enumerate(self.agent_ids):
            progress_delta = torch.clamp(
                self.agent_guidance_progress[agent] - self.agent_prev_guidance_progress[agent],
                min=0.0,
                max=self.cfg.guidance_progress_clamp,
            )
            progress_reward = (
                progress_delta / max(self.cfg.guidance_progress_clamp, 1e-6)
            ) * torch.exp(
                -torch.square(self.agent_guidance_lateral_error[agent] / max(self.cfg.guidance_lateral_sigma, 1e-6))
            )
            wrong_way_penalty = torch.clamp(
                self.agent_prev_guidance_progress[agent] - self.agent_guidance_progress[agent],
                min=0.0,
                max=self.cfg.guidance_wrong_way_clamp,
            ) / max(self.cfg.guidance_wrong_way_clamp, 1e-6)
            lateral_penalty = 1.0 - torch.exp(
                -0.5 * torch.square(self.agent_guidance_lateral_error[agent] / max(self.cfg.guidance_lateral_sigma, 1e-6))
            )
            goal_distance = self.agent_goal_distance[agent]
            goal_soft_reward = 1.0 / (1.0 + torch.square(goal_distance / self.cfg.goal_soft_sigma))
            goal_tight_reward = 1.0 / (1.0 + torch.square(goal_distance / self.cfg.goal_tight_sigma))
            cluster_goal_bonus = 1.0 / (
                1.0 + torch.square(self.cluster_goal_distance / max(self.cfg.cluster_goal_bonus_sigma, 1e-6))
            )

            close_penalty = torch.clamp(
                (self.cfg.min_separation - nearest_distances[:, agent_idx]) / self.cfg.min_separation,
                min=0.0,
                max=1.0,
            )
            far_penalty = torch.clamp(
                (centroid_spread[:, agent_idx] - self.cfg.max_cohesion_radius) / self.cfg.max_cohesion_radius,
                min=0.0,
                max=1.0,
            )
            pairwise_far_penalty = torch.clamp(
                (max_pairwise_distance - self.cfg.max_pairwise_separation) / self.cfg.max_pairwise_separation,
                min=0.0,
                max=1.0,
            )
            speed_xy = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w[:, :2], dim=1)
            overspeed_penalty = torch.clamp((speed_xy - self.cfg.max_speed) / self.cfg.max_speed, min=0.0)
            action_rate_penalty = torch.sum(torch.abs(self.actions[agent] - self._previous_actions[agent]), dim=1)
            termination_penalty = (self.cluster_collision.float() + self.cluster_contact.float()).clamp(max=1.0)

            rewards[agent] = (
                self.cfg.reward_progress_weight * progress_reward
                - self.cfg.reward_wrong_way_weight * wrong_way_penalty
                - self.cfg.reward_lateral_weight * lateral_penalty
                + self.cfg.reward_goal_soft_weight * goal_soft_reward
                + self.cfg.reward_goal_tight_weight * goal_tight_reward
                + self.cfg.reward_cluster_goal_bonus_weight * cluster_goal_bonus
                + self.cfg.reward_enter_target_region_weight * enter_target_region_bonus.float()
                + self.cfg.reward_success_weight * self.cluster_success.float()
                - self.cfg.reward_close_weight * close_penalty
                - self.cfg.reward_far_weight * far_penalty
                - self.cfg.reward_pairwise_far_weight * pairwise_far_penalty
                - self.cfg.reward_collision_weight * self.cluster_collision.float()
                - self.cfg.reward_contact_weight * self.agent_contact_penalty_active[agent].float()
                - self.cfg.reward_termination_weight * termination_penalty
                - self.cfg.reward_overspeed_weight * overspeed_penalty
                - self.cfg.reward_action_rate_weight * action_rate_penalty
            )
        self.cluster_target_region_bonus_awarded |= self.cluster_target_region_reached
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._update_common_buffers()
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated_common = self.cluster_success | self.cluster_collision | self.cluster_contact
        terminal_mask = terminated_common | time_out
        self._pending_log = self._build_terminal_log(terminal_mask, time_out)
        terminated = {agent: terminated_common.clone() for agent in self.agent_ids}
        time_outs = {agent: time_out.clone() for agent in self.agent_ids}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.scene._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for agent in self.agent_ids:
            self.robots[agent].reset(env_ids)
            self.contact_sensors[agent].reset(env_ids)
            self.cameras[agent].reset(env_ids)
            self.height_scanners[agent].reset(env_ids)
        super()._reset_idx(env_ids)

        for agent in self.agent_ids:
            self.actions[agent][env_ids] = 0.0
            self._processed_actions[agent][env_ids] = 0.0
            self._previous_actions[agent][env_ids] = 0.0
            self._desired_velocity_w[agent][env_ids] = 0.0

        (
            source_region_ids,
            target_region_ids,
            guidance_ids,
            spawn_centers,
            goal_centers,
            spawn_positions,
        ) = self._sample_spawn_and_goal(env_ids)
        self.source_region_ids[env_ids] = source_region_ids
        self.target_region_ids[env_ids] = target_region_ids
        self.current_guidance_ids[env_ids] = guidance_ids
        self.cluster_spawn_center[env_ids] = spawn_centers
        self.cluster_goal_center[env_ids] = goal_centers
        self.cluster_centroid[env_ids] = spawn_positions.mean(dim=1)
        self.cluster_success[env_ids] = False
        self.cluster_collision[env_ids] = False
        self.cluster_contact[env_ids] = False
        self.cluster_fall[env_ids] = False
        self.cluster_goal_distance[env_ids] = torch.linalg.norm(
            self.cluster_goal_center[env_ids, :2] - self.cluster_centroid[env_ids, :2], dim=1
        )
        self.cluster_max_centroid_radius[env_ids] = 0.0
        self.cluster_max_pairwise_distance[env_ids] = 0.0
        self.cluster_target_region_reached[env_ids] = False
        self.cluster_target_region_bonus_awarded[env_ids] = False

        for agent_idx, agent in enumerate(self.agent_ids):
            robot = self.robots[agent]
            root_state = robot.data.default_root_state[env_ids].clone()
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            yaw = torch.rand((len(env_ids),), device=self.device) * (2.0 * math.pi) - math.pi
            root_state[:, :3] = spawn_positions[:, agent_idx]
            root_state[:, 3:7] = quat_from_euler_xyz(
                torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
            )
            root_state[:, 7:] = 0.0
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            initial_progress, initial_lateral_error = self._project_guidance_state(
                spawn_positions[:, agent_idx, :2], guidance_ids
            )
            self.agent_guidance_progress[agent][env_ids] = initial_progress
            self.agent_prev_guidance_progress[agent][env_ids] = initial_progress
            self.agent_guidance_lateral_error[agent][env_ids] = initial_lateral_error
            self.agent_goal_distance[agent][env_ids] = torch.linalg.norm(
                self.cluster_goal_center[env_ids, :2] - spawn_positions[:, agent_idx, :2], dim=1
            )
            self.agent_contact_penalty_active[agent][env_ids] = False
            self.agent_contact_termination_streak[agent][env_ids] = 0

    def _update_common_buffers(self) -> None:
        positions = torch.stack([self.robots[agent].data.root_pos_w for agent in self.agent_ids], dim=1)
        self.cluster_centroid = positions.mean(dim=1)
        self.cluster_goal_distance = torch.linalg.norm(self.cluster_goal_center[:, :2] - self.cluster_centroid[:, :2], dim=1)

        pairwise_distances = self._compute_pairwise_distances_xy()
        centroid_spread = self._compute_centroid_spread()
        collision_matrix = (pairwise_distances > 0.0) & (pairwise_distances < self.cfg.collision_distance)
        self.cluster_collision = collision_matrix.any(dim=(1, 2))
        self.cluster_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_fall = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cluster_max_centroid_radius = centroid_spread.amax(dim=1)
        self.cluster_max_pairwise_distance = pairwise_distances.amax(dim=(1, 2))
        agent_goal_distances = []
        for agent in self.agent_ids:
            robot = self.robots[agent]
            self.agent_prev_guidance_progress[agent].copy_(self.agent_guidance_progress[agent])
            progress, lateral_error = self._project_guidance_state(robot.data.root_pos_w[:, :2], self.current_guidance_ids)
            self.agent_guidance_progress[agent].copy_(progress)
            self.agent_guidance_lateral_error[agent].copy_(lateral_error)
            self.agent_goal_distance[agent] = torch.linalg.norm(
                self.cluster_goal_center[:, :2] - robot.data.root_pos_w[:, :2], dim=1
            )

            contact_force = self._compute_contact_force(agent)
            self.agent_contact_penalty_active[agent] = contact_force > self.cfg.contact_force_threshold
            termination_contact_active = contact_force > self.cfg.contact_termination_force_threshold
            self.agent_contact_termination_streak[agent] = torch.where(
                termination_contact_active,
                self.agent_contact_termination_streak[agent] + 1,
                torch.zeros_like(self.agent_contact_termination_streak[agent]),
            )
            self.cluster_contact |= self.agent_contact_termination_streak[agent] >= self.cfg.contact_termination_steps
            agent_goal_distances.append(self.agent_goal_distance[agent])

        stacked_goal_distances = torch.stack(agent_goal_distances, dim=1)
        self.cluster_target_region_reached = (self.cluster_goal_distance < self.cfg.cluster_entry_radius) & (
            torch.max(stacked_goal_distances, dim=1).values < self.cfg.agent_entry_radius
        )
        success_goal = (self.cluster_goal_distance < self.cfg.cluster_success_radius) & (
            torch.max(stacked_goal_distances, dim=1).values < self.cfg.agent_goal_radius
        )
        success_compact = (self.cluster_max_centroid_radius < self.cfg.success_max_centroid_radius) & (
            self.cluster_max_pairwise_distance < self.cfg.success_max_pairwise_distance
        )
        self.cluster_success = success_goal & success_compact

        mean_progress_delta = torch.stack(
            [self.agent_guidance_progress[a] - self.agent_prev_guidance_progress[a] for a in self.agent_ids],
            dim=1,
        ).mean()
        mean_lateral_error = torch.stack(
            [self.agent_guidance_lateral_error[a] for a in self.agent_ids],
            dim=1,
        ).mean()
        mean_spread = centroid_spread.mean()
        for agent in self.agent_ids:
            self.extras[agent].pop("log", None)
            self.extras[agent]["metrics"] = {
                "cluster_success_rate": self.cluster_success.float().mean(),
                "cluster_collision_rate": self.cluster_collision.float().mean(),
                "cluster_contact_rate": self.cluster_contact.float().mean(),
                "cluster_target_region_rate": self.cluster_target_region_reached.float().mean(),
                "cluster_goal_distance": self.cluster_goal_distance.mean(),
                "cluster_spread": mean_spread,
                "cluster_max_centroid_radius": self.cluster_max_centroid_radius.mean(),
                "cluster_max_pairwise_distance": self.cluster_max_pairwise_distance.mean(),
                "guidance_progress_delta": mean_progress_delta,
                "guidance_lateral_error": mean_lateral_error,
            }

    def _build_terminal_log(self, terminal_mask: torch.Tensor, time_out: torch.Tensor) -> dict[str, torch.Tensor] | None:
        if not torch.any(terminal_mask):
            return None

        terminal_success = self.cluster_success & terminal_mask
        terminal_collision = self.cluster_collision & terminal_mask
        terminal_contact = self.cluster_contact & terminal_mask
        terminal_timeout = time_out & terminal_mask
        terminal_target_region = self.cluster_target_region_reached & terminal_mask
        terminal_count = terminal_mask.sum()
        terminal_mask_f = terminal_mask.float()

        mean_lateral_error = torch.stack(
            [self.agent_guidance_lateral_error[a] for a in self.agent_ids],
            dim=1,
        ).mean(dim=1)
        mean_goal_distance = torch.stack(
            [self.agent_goal_distance[a] for a in self.agent_ids],
            dim=1,
        ).mean(dim=1)

        def masked_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * terminal_mask_f).sum() / terminal_count.float()

        return {
            "terminal_count": terminal_count.float(),
            "terminal_success_rate": terminal_success.float().sum() / terminal_count.float(),
            "terminal_collision_rate": terminal_collision.float().sum() / terminal_count.float(),
            "terminal_contact_rate": terminal_contact.float().sum() / terminal_count.float(),
            "terminal_target_region_rate": terminal_target_region.float().sum() / terminal_count.float(),
            "terminal_timeout_rate": terminal_timeout.float().sum() / terminal_count.float(),
            "terminal_goal_distance": masked_mean(self.cluster_goal_distance),
            "terminal_mean_agent_goal_distance": masked_mean(mean_goal_distance),
            "terminal_max_centroid_radius": masked_mean(self.cluster_max_centroid_radius),
            "terminal_max_pairwise_distance": masked_mean(self.cluster_max_pairwise_distance),
            "terminal_guidance_lateral_error": masked_mean(mean_lateral_error),
        }

    def _sample_spawn_and_goal(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_indices = torch.zeros((len(env_ids),), dtype=torch.long, device=self.device)
        source_region_ids = torch.zeros((len(env_ids),), dtype=torch.long, device=self.device)
        target_region_ids = torch.zeros((len(env_ids),), dtype=torch.long, device=self.device)

        spawn_centers = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        goal_centers = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        spawn_positions = torch.zeros(
            (len(env_ids), self.num_agents_cfg, 3), dtype=torch.float32, device=self.device
        )

        for local_idx in range(len(env_ids)):
            pair_order = torch.randperm(len(self.guidance_pair_ids), device=self.device)
            sampled = False
            for pair_idx in pair_order.tolist():
                source_region_id = int(self.guidance_pair_ids[pair_idx, 0].item())
                target_region_id = int(self.guidance_pair_ids[pair_idx, 1].item())
                region_points = self.region_safe_points[source_region_id]
                goal_region_points = self.region_safe_points[target_region_id]
                if len(goal_region_points) == 0:
                    continue
                cluster_sample = self._sample_cluster_points(region_points, source_region_id)
                if cluster_sample is None:
                    continue

                cluster_points, center_point = cluster_sample
                goal_idx = torch.randint(0, len(goal_region_points), (1,), device=self.device).item()
                goal_centers[local_idx] = goal_region_points[goal_idx]
                spawn_positions[local_idx] = cluster_points
                spawn_centers[local_idx] = center_point
                sampled_indices[local_idx] = pair_idx
                source_region_ids[local_idx] = source_region_id
                target_region_ids[local_idx] = target_region_id
                sampled = True
                break

            if not sampled:
                raise RuntimeError("Unable to sample a legal clustered five-drone spawn from any configured region pair.")

        return source_region_ids, target_region_ids, sampled_indices, spawn_centers, goal_centers, spawn_positions

    def _sample_cluster_points(
        self, region_points: torch.Tensor, source_region_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(region_points) < self.num_agents_cfg:
            return None

        for _ in range(self.cfg.cluster_sampling_attempts):
            center_idx = torch.randint(0, len(region_points), (1,), device=self.device).item()
            center_point = region_points[center_idx]
            yaw = torch.rand(1, device=self.device).item() * (2.0 * math.pi)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            rotation = torch.tensor([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=torch.float32, device=self.device)
            for scale in self.cfg.initial_formation_scales:
                rotated_offsets = (self._formation_offsets * scale) @ rotation.T
                selected_indices: list[int] = []
                selected_points: list[torch.Tensor] = []
                success = True
                for offset in rotated_offsets:
                    target_xy = center_point[:2] + offset
                    distances = torch.linalg.norm(region_points[:, :2] - target_xy.unsqueeze(0), dim=1)
                    candidate_indices = torch.argsort(distances)
                    picked = False
                    for idx in candidate_indices[: self.cfg.spawn_candidate_pool]:
                        if distances[idx] > self.cfg.spawn_assignment_radius:
                            break
                        if int(idx.item()) in selected_indices:
                            continue
                        candidate = region_points[idx]
                        if selected_points:
                            pairwise = torch.stack(
                                [torch.linalg.norm(candidate[:2] - point[:2]) for point in selected_points]
                            )
                            if torch.any(pairwise < self.cfg.min_spawn_separation):
                                continue
                        selected_indices.append(int(idx.item()))
                        selected_points.append(candidate)
                        picked = True
                        break
                    if not picked:
                        success = False
                        break
                if success and len(selected_points) == self.num_agents_cfg:
                    return torch.stack(selected_points, dim=0), center_point

        candidate_centers = self.region_fallback_center_indices[source_region_id]
        if len(candidate_centers) == 0:
            return None

        permutation = torch.randperm(len(candidate_centers), device=self.device)
        for perm_idx in permutation.tolist():
            idx = int(candidate_centers[perm_idx].item())
            center_point = region_points[idx]
            candidate_points = self._select_cluster_points_near_center(
                region_points=region_points,
                center_point=center_point,
                max_radius=self.cfg.fallback_cluster_radius,
            )
            if candidate_points is not None and len(candidate_points) == self.num_agents_cfg:
                return candidate_points, center_point

        return None

    def _project_guidance_state(
        self, positions_xy: torch.Tensor, guidance_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _compute_pairwise_distances_xy(self) -> torch.Tensor:
        positions = torch.stack([self.robots[agent].data.root_pos_w[:, :2] for agent in self.agent_ids], dim=1)
        return torch.cdist(positions, positions)

    def _compute_nearest_neighbor_distances(self, pairwise_distances: torch.Tensor) -> torch.Tensor:
        masked = pairwise_distances + torch.eye(self.num_agents_cfg, device=self.device).unsqueeze(0) * 1e6
        return torch.min(masked, dim=2).values

    def _compute_centroid_spread(self) -> torch.Tensor:
        positions = torch.stack([self.robots[agent].data.root_pos_w[:, :2] for agent in self.agent_ids], dim=1)
        centroid = positions.mean(dim=1, keepdim=True)
        return torch.linalg.norm(positions - centroid, dim=2)

    def _compute_contact_penalty(self, agent: str) -> torch.Tensor:
        return (self._compute_contact_force(agent) > self.cfg.contact_force_threshold).float()

    def _compute_contact_force(self, agent: str) -> torch.Tensor:
        sensor = self.contact_sensors[agent]
        body_ids = self.contact_body_ids[agent]
        net_contact_forces = sensor.data.net_forces_w_history
        body_index = body_ids if isinstance(body_ids, list) else [body_ids]
        contact_force = torch.max(torch.norm(net_contact_forces[:, :, body_index], dim=-1), dim=1)[0]
        return torch.max(contact_force, dim=1).values

    def _select_cluster_points_near_center(
        self, region_points: torch.Tensor, center_point: torch.Tensor, max_radius: float
    ) -> torch.Tensor | None:
        distances = torch.linalg.norm(region_points[:, :2] - center_point[:2].unsqueeze(0), dim=1)
        candidate_indices = torch.argsort(distances)
        selected_points: list[torch.Tensor] = []
        for idx in candidate_indices.tolist():
            if float(distances[idx].item()) > max_radius:
                break
            candidate = region_points[idx]
            if selected_points:
                pairwise = torch.stack([torch.linalg.norm(candidate[:2] - point[:2]) for point in selected_points])
                if torch.any(pairwise < self.cfg.min_spawn_separation):
                    continue
            selected_points.append(candidate)
            if len(selected_points) == self.num_agents_cfg:
                selected_stack = torch.stack(selected_points, dim=0)
                pairwise = torch.cdist(selected_stack[:, :2], selected_stack[:, :2])
                if torch.amax(pairwise) <= self.cfg.fallback_cluster_pairwise_distance:
                    return selected_stack
                selected_points.pop()
        if not selected_points:
            return None
        return None

    def _build_region_fallback_center_indices(self) -> list[torch.Tensor]:
        fallback_center_indices: list[torch.Tensor] = []
        for region_points in self.region_safe_points:
            feasible_centers: list[int] = []
            for center_idx in range(len(region_points)):
                candidate_points = self._select_cluster_points_near_center(
                    region_points=region_points,
                    center_point=region_points[center_idx],
                    max_radius=self.cfg.fallback_cluster_radius,
                )
                if candidate_points is not None and len(candidate_points) == self.num_agents_cfg:
                    feasible_centers.append(center_idx)

            fallback_center_indices.append(
                torch.tensor(feasible_centers, dtype=torch.long, device=self.device)
            )
        return fallback_center_indices

    def _get_depth_features(self, agent: str) -> torch.Tensor:
        return depth_image_prefect(self, SceneEntityCfg(f"raycast_camera_{self.agent_ids.index(agent)}"))

    def _get_height_features(self, agent: str) -> torch.Tensor:
        return height_scan_feat(self, SceneEntityCfg(f"height_scanner_critic_{self.agent_ids.index(agent)}"))
