from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .drone_accel_actions import DroneAccelAction


@configclass
class DroneAccelActionCfg(ActionTermCfg):
    """Configuration for planar drone acceleration commands."""

    class_type: type[ActionTerm] = DroneAccelAction

    asset_name: str = MISSING
    scale: list[float] = [1.0, 1.0, 1.5]
    offset: list[float] = [0.0, 0.0, 0.0]
    use_raw_actions: bool = False
    policy_distr_type: str = "gaussian"
    target_height: float = 1.2
    body_name: str = "body"
    max_acceleration: float = 1.0
    max_speed: float = 1.0
    use_controller: bool = True
    controller_decimation: int = 2
    controller_k_max_ang: float = 30.0
    enable_execution_delay: bool = True
    execution_delay_steps_range: tuple[int, int] = (4, 8)
    enable_action_lag: bool = True
    action_lag_time_constant_range_s: tuple[float, float] = (0.06, 0.10)
    execution_scale_range_xy: tuple[float, float] = (0.95, 1.05)
    execution_scale_range_yaw: tuple[float, float] = (0.95, 1.05)
    execution_bias_range_xy: tuple[float, float] = (-0.03, 0.03)
    execution_bias_range_yaw: tuple[float, float] = (-0.03, 0.03)
    execution_noise_range_xy: tuple[float, float] = (-0.05, 0.05)
    execution_noise_range_yaw: tuple[float, float] = (-0.03, 0.03)
