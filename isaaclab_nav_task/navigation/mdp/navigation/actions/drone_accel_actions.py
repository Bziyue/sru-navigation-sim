"""SE2 action term for a drone constrained to planar flight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat, yaw_quat
from isaaclab_nav_task.navigation.utils.controller import Controller

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .drone_accel_actions_cfg import DroneAccelActionCfg


class DroneAccelAction(ActionTerm):
    """Map current-body-frame `[ax, ay, yaw_rate]` commands into planar drone control."""

    cfg: DroneAccelActionCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: DroneAccelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._action_dim = 3
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._preclipped_actions = torch.zeros_like(self._raw_actions)
        self._root_twist_command = torch.zeros((self.num_envs, 6), device=self.device)
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._physics_dt = self._env.physics_dt
        self._desired_velocity_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_acceleration_w = torch.zeros_like(self._desired_velocity_w)
        self._desired_position_w = torch.zeros_like(self._desired_velocity_w)
        self._desired_jerk_w = torch.zeros_like(self._desired_velocity_w)
        self._desired_yaw = torch.zeros((self.num_envs, 1), device=self.device)
        self._desired_yaw_rate = torch.zeros((self.num_envs, 1), device=self.device)
        self._yaw_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._controller_counter = 0

        if self.cfg.use_controller:
            self._body_id = self._asset.find_bodies(self.cfg.body_name)[0]
            self._root_force_command = torch.zeros((self.num_envs, 1, 3), device=self.device)
            self._root_torque_command = torch.zeros((self.num_envs, 1, 3), device=self.device)
            gravity = torch.tensor(self._env.sim.cfg.gravity, device=self.device, dtype=torch.float32)
            mass = self._asset.root_physx_view.get_masses()[0, 0].to(device=self.device, dtype=torch.float32)
            inertia = self._asset.root_physx_view.get_inertias()[0, 0].to(device=self.device, dtype=torch.float32)
            controller_dt = self.cfg.controller_decimation * self._physics_dt
            self._controller = Controller(
                step_dt=controller_dt,
                gravity=gravity,
                mass=mass,
                inertia=inertia,
                num_envs=self.num_envs,
                k_max_ang=self.cfg.controller_k_max_ang,
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def preclipped_actions(self) -> torch.Tensor:
        return self._preclipped_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.cfg.use_raw_actions:
            self._processed_actions[:] = self._raw_actions
        else:
            self._processed_actions[:] = self._raw_actions * self._scale + self._offset

        if self.cfg.policy_distr_type == "gaussian":
            self._processed_actions[:] = torch.tanh(self._processed_actions)
        elif self.cfg.policy_distr_type == "beta":
            self._processed_actions[:] = (self._processed_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        self._processed_actions[:] = self._processed_actions * self._scale
        self._preclipped_actions[:] = self._processed_actions
        self._processed_actions[:, :2] = torch.clamp(
            self._processed_actions[:, :2],
            min=-self.cfg.max_acceleration,
            max=self.cfg.max_acceleration,
        )
        self._desired_velocity_w.zero_()
        self._desired_velocity_w[:, :2] = self._asset.data.root_lin_vel_w[:, :2]
        self._clip_planar_speed_(self._desired_velocity_w)

    def _clip_planar_speed_(self, planar_velocity_w: torch.Tensor) -> None:
        speed_xy = torch.linalg.norm(planar_velocity_w[:, :2], dim=1, keepdim=True)
        clip_scale = torch.clamp(speed_xy / self.cfg.max_speed, min=1.0)
        planar_velocity_w[:, :2] /= clip_scale

    def _compute_planar_acceleration_world(self) -> torch.Tensor:
        yaw_only_quat = yaw_quat(self._asset.data.root_quat_w)
        planar_acc_body = torch.zeros((self.num_envs, 3), device=self.device)
        planar_acc_body[:, :2] = self._processed_actions[:, :2]
        planar_acc_world = math_utils.quat_apply(yaw_only_quat, planar_acc_body)
        planar_acc_world[:, :2] = torch.clamp(
            planar_acc_world[:, :2],
            min=-self.cfg.max_acceleration,
            max=self.cfg.max_acceleration,
        )
        return planar_acc_world

    def _initialize_desired_yaw(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.nonzero(~self._yaw_initialized, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        root_yaw_quat = yaw_quat(self._asset.data.root_quat_w[env_ids])
        root_yaw = euler_xyz_from_quat(root_yaw_quat)[2].unsqueeze(1)
        self._desired_yaw[env_ids] = root_yaw
        self._yaw_initialized[env_ids] = True

    def _advance_desired_motion(self) -> None:
        self._initialize_desired_yaw()
        self._desired_acceleration_w.zero_()
        self._desired_acceleration_w[:] = self._compute_planar_acceleration_world()
        self._desired_velocity_w[:, :2] += self._desired_acceleration_w[:, :2] * self._physics_dt
        self._clip_planar_speed_(self._desired_velocity_w)
        self._desired_yaw_rate[:] = self._processed_actions[:, 2:3]
        self._desired_yaw[:] = self._desired_yaw + self._desired_yaw_rate * self._physics_dt

    def apply_actions(self):
        self._advance_desired_motion()
        if self.cfg.use_controller:
            self._apply_controller_actions()
        else:
            self._apply_ideal_actions()

    def _apply_ideal_actions(self) -> None:
        root_pos = self._asset.data.root_pos_w.clone()
        yaw_only_quat = yaw_quat(self._asset.data.root_quat_w)

        pose = torch.cat((root_pos, yaw_only_quat), dim=-1)
        pose[:, 2] = self.cfg.target_height
        self._asset.write_root_pose_to_sim(pose)

        self._root_twist_command[:, 0:2] = self._desired_velocity_w[:, 0:2]
        self._root_twist_command[:, 2] = 0.0
        self._root_twist_command[:, 3:5] = 0.0
        self._root_twist_command[:, 5] = self._desired_yaw_rate[:, 0]
        self._asset.write_root_velocity_to_sim(self._root_twist_command)

    def _apply_controller_actions(self) -> None:
        root_pos = self._asset.data.root_pos_w

        self._desired_position_w[:] = root_pos
        self._desired_position_w[:, 2] = self.cfg.target_height

        if self._controller_counter % self.cfg.controller_decimation == 0:
            desired_state = torch.cat(
                (
                    self._desired_position_w,
                    self._desired_velocity_w,
                    self._desired_acceleration_w,
                    self._desired_jerk_w,
                    self._desired_yaw,
                    self._desired_yaw_rate,
                ),
                dim=1,
            )
            _, thrust, _, _, torque = self._controller.get_control(
                self._asset.data.root_state_w,
                desired_state,
            )
            self._root_force_command.zero_()
            self._root_force_command[:, 0, 2] = thrust
            self._root_torque_command[:, 0, :] = torque
            self._controller_counter = 0

        self._controller_counter += 1
        self._asset.set_external_force_and_torque(
            self._root_force_command,
            self._root_torque_command,
            body_ids=self._body_id,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._preclipped_actions.zero_()
            self._root_twist_command.zero_()
            self._desired_velocity_w.zero_()
            self._desired_acceleration_w.zero_()
            self._desired_position_w.zero_()
            self._desired_jerk_w.zero_()
            self._desired_yaw.zero_()
            self._desired_yaw_rate.zero_()
            self._yaw_initialized.zero_()
            self._controller_counter = 0
            if self.cfg.use_controller:
                self._root_force_command.zero_()
                self._root_torque_command.zero_()
                self._controller.reset()
            return

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._preclipped_actions[env_ids] = 0.0
        self._root_twist_command[env_ids] = 0.0
        self._desired_velocity_w[env_ids] = 0.0
        self._desired_acceleration_w[env_ids] = 0.0
        self._desired_position_w[env_ids] = 0.0
        self._desired_jerk_w[env_ids] = 0.0
        self._desired_yaw[env_ids] = 0.0
        self._desired_yaw_rate[env_ids] = 0.0
        self._yaw_initialized[env_ids] = False
        if self.cfg.use_controller:
            self._root_force_command[env_ids] = 0.0
            self._root_torque_command[env_ids] = 0.0
            self._controller.reset(env_ids)
