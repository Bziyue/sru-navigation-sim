"""SE2 action term for a drone constrained to planar flight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .drone_se2_actions_cfg import DroneSE2ActionCfg


class DroneSE2Action(ActionTerm):
    """Map `[vx, vy, yaw_rate]` commands into planar drone root control."""

    cfg: DroneSE2ActionCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: DroneSE2ActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._action_dim = 3
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._root_velocity_command = torch.zeros((self.num_envs, 6), device=self.device)
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

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

    def apply_actions(self):
        root_pos = self._asset.data.root_pos_w.clone()
        root_quat = self._asset.data.root_quat_w
        yaw_only_quat = yaw_quat(root_quat)

        pose = torch.cat((root_pos, yaw_only_quat), dim=-1)
        pose[:, 2] = self.cfg.target_height
        self._asset.write_root_pose_to_sim(pose)

        yaw = euler_xyz_from_quat(yaw_only_quat)[2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_body = self._processed_actions[:, 0]
        vy_body = self._processed_actions[:, 1]

        self._root_velocity_command[:, 0] = cos_yaw * vx_body - sin_yaw * vy_body
        self._root_velocity_command[:, 1] = sin_yaw * vx_body + cos_yaw * vy_body
        self._root_velocity_command[:, 2] = 0.0
        self._root_velocity_command[:, 3:5] = 0.0
        self._root_velocity_command[:, 5] = self._processed_actions[:, 2]

        self._asset.write_root_velocity_to_sim(self._root_velocity_command)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._root_velocity_command[env_ids] = 0.0
