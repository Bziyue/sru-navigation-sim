"""Configuration for static region-to-region drone swarm navigation with shared SRU policy."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_nav_task.navigation.assets import DJI_FPV_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR


PLANNING_FREQ = 5.0
STATIC_SCAN_DIR = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Environments", "StaticScan")
STATIC_VISUAL_MESH_PRIM_PATH = "/World/StaticMesh"
STATIC_COLLISION_MESH_PRIM_PATH = "/World/MapMesh"

AGENT_IDS = ["drone_0", "drone_1", "drone_2"]
DEPTH_FEATURE_DIM = 64 * 5 * 8
HEIGHT_FEATURE_DIM = 64 * 7 * 7
POLICY_PROPRIO_DIM = 16 + (len(AGENT_IDS) - 1) * 5
CRITIC_PROPRIO_DIM = POLICY_PROPRIO_DIM
POLICY_OBS_DIM = POLICY_PROPRIO_DIM + DEPTH_FEATURE_DIM
CRITIC_OBS_DIM = CRITIC_PROPRIO_DIM + 1 + HEIGHT_FEATURE_DIM + DEPTH_FEATURE_DIM


@configclass
class DroneSwarmStaticSceneCfg(InteractiveSceneCfg):
    robot_0: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_0")
    robot_1: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    robot_2: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")

    static_mesh = AssetBaseCfg(
        prim_path=STATIC_VISUAL_MESH_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(STATIC_SCAN_DIR, "DR_static_mesh.usdc"),
        ),
    )

    raycast_camera_0 = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/body",
        mesh_prim_paths=[STATIC_COLLISION_MESH_PRIM_PATH],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.02),
            rot=(0.9848078, 0.0, 0.1736482, 0.0),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        max_distance=11.0,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            fx=72.7025,
            fy=72.7025,
            cx=94.4457,
            cy=62.5424,
            width=192,
            height=120,
            downsample_factor=3,
        ),
    )
    raycast_camera_1 = raycast_camera_0.replace(prim_path="{ENV_REGEX_NS}/Robot_1/body")
    raycast_camera_2 = raycast_camera_0.replace(prim_path="{ENV_REGEX_NS}/Robot_2/body")

    height_scanner_critic_0 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
        debug_vis=False,
        mesh_prim_paths=[STATIC_COLLISION_MESH_PRIM_PATH],
    )
    height_scanner_critic_1 = height_scanner_critic_0.replace(prim_path="{ENV_REGEX_NS}/Robot_1/body")
    height_scanner_critic_2 = height_scanner_critic_0.replace(prim_path="{ENV_REGEX_NS}/Robot_2/body")

    contact_forces_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/body",
        history_length=3,
        track_air_time=True,
    )
    contact_forces_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/body",
        history_length=3,
        track_air_time=True,
    )
    contact_forces_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/body",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class DroneSwarmStaticNavigationEnvCfg(DirectMARLEnvCfg):
    possible_agents = AGENT_IDS
    use_parameter_sharing_wrapper: bool = True

    scene: DroneSwarmStaticSceneCfg = DroneSwarmStaticSceneCfg(
        num_envs=128,
        env_spacing=24.0,
        replicate_physics=False,
    )

    episode_length_s = 60.0
    decimation = int((1.0 / 0.005) / PLANNING_FREQ)
    is_finite_horizon = True
    debug_vis = False

    observation_spaces = {
        agent: {"policy": POLICY_OBS_DIM, "critic": CRITIC_OBS_DIM}
        for agent in AGENT_IDS
    }
    state_space = 0
    action_spaces = {agent: 3 for agent in AGENT_IDS}

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=decimation,
    )

    num_agents: int = len(AGENT_IDS)
    static_visual_mesh_prim_path: str = STATIC_VISUAL_MESH_PRIM_PATH
    static_collision_mesh_prim_path: str = STATIC_COLLISION_MESH_PRIM_PATH
    boundary_wall_padding: float = 2.0

    surface_bbox_data_path: str = os.path.join(STATIC_SCAN_DIR, "DR_Surface_BBox_Data.txt")
    guidance_trajectories_data_path: str = os.path.join(STATIC_SCAN_DIR, "all_region_pair_trajectories.json")

    flight_height: float = 1.2
    point_clearance: float = 0.15
    safe_point_grid_spacing: float = 0.25
    guidance_trajectory_eval_dt: float = 0.05
    guidance_arc_length_spacing: float = 0.2

    action_scale_xy: float = 2.5
    action_scale_yaw: float = 1.5
    max_agent_distance_policy: float = 3.0
    max_agent_distance_critic: float = 5.0

    agent_spawn_separation: float = 0.6
    agent_goal_separation: float = 0.6
    agent_collision_distance: float = 0.35
    agent_separation_distance: float = 1.0
    body_contact_force_threshold: float = 0.01
    fall_height_threshold: float = 0.5
    goal_completion_radius: float = 0.45
    required_goal_hold_time_s: float = 1.0
    soft_goal_radius: float = 1.0
    tight_goal_radius: float = 0.25

    guidance_progress_clamp: float = 0.3
    guidance_wrong_way_clamp: float = 0.2
    guidance_corridor_sigma: float = 0.6
    guidance_lateral_sigma: float = 0.6

    reward_action_rate_l1: float = 0.05
    reward_guidance_progress: float = 0.7
    reward_guidance_wrong_way: float = 0.3
    reward_guidance_lateral_error: float = 0.25
    reward_reach_goal_soft: float = 0.25
    reward_reach_goal_tight: float = 1.5
    reward_agent_collision: float = 20.0
    reward_agent_separation: float = 0.5
    reward_team_success: float = 5.0
    reward_episode_failure: float = 50.0

    def __post_init__(self):
        self.sim.dt = 0.005
        self.is_finite_horizon = True
        self.decimation = int((1.0 / self.sim.dt) / PLANNING_FREQ)
        self.episode_length_s = 60.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False

        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config
        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        initialize_depth_noise_generator(
            camera_config=get_camera_config("b2w", use_default_fallback=True),
            use_jit_precompiled=False,
        )

        for sensor in [
            self.scene.raycast_camera_0,
            self.scene.raycast_camera_1,
            self.scene.raycast_camera_2,
        ]:
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt

        for sensor in [
            self.scene.height_scanner_critic_0,
            self.scene.height_scanner_critic_1,
            self.scene.height_scanner_critic_2,
        ]:
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt

        for sensor in [
            self.scene.contact_forces_0,
            self.scene.contact_forces_1,
            self.scene.contact_forces_2,
        ]:
            if sensor is not None:
                sensor.update_period = self.sim.dt


@configclass
class DroneSwarmStaticNavigationEnvCfg_DEV(DroneSwarmStaticNavigationEnvCfg):
    scene: DroneSwarmStaticSceneCfg = DroneSwarmStaticSceneCfg(
        num_envs=24,
        env_spacing=24.0,
        replicate_physics=False,
    )


@configclass
class DroneSwarmStaticNavigationEnvCfg_PLAY(DroneSwarmStaticNavigationEnvCfg):
    scene: DroneSwarmStaticSceneCfg = DroneSwarmStaticSceneCfg(
        num_envs=1,
        env_spacing=24.0,
        replicate_physics=False,
    )
