from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .drone_se2_actions import DroneSE2Action


@configclass
class DroneSE2ActionCfg(ActionTermCfg):
    """Configuration for planar drone velocity commands."""

    class_type: type[ActionTerm] = DroneSE2Action

    asset_name: str = MISSING
    scale: list[float] = [2.0, 2.0, 1.5]
    offset: list[float] = [0.0, 0.0, 0.0]
    use_raw_actions: bool = False
    policy_distr_type: str = "gaussian"
    target_height: float = 1.2
    body_name: str = "body"
