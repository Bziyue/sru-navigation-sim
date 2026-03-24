"""Configuration for 5-drone swarm navigation with shared-parameter IPPO on a static scanned mesh."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_nav_task.navigation.assets import DJI_FPV_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR
from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config
from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator


PLANNING_FREQ = 5.0
STATIC_SCAN_DIR = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Environments", "StaticScan")
STATIC_VISUAL_MESH_PRIM_PATH = "/World/StaticMesh"
STATIC_COLLISION_MESH_PRIM_PATH = "/World/MapMesh"
DEFAULT_PRECOMPUTED_SAFE_POINTS_PATH = os.path.join(
    STATIC_SCAN_DIR,
    "DR_region_safe_points_contact_0p2m_1p2_to_2p0_eroded_0p4m.npz",
)

NUM_AGENTS = 5
TEAMMATE_FEATURE_DIM = 6 * (NUM_AGENTS - 1)
POLICY_PROPRIO_DIM = 16 + TEAMMATE_FEATURE_DIM
CRITIC_PROPRIO_DIM = POLICY_PROPRIO_DIM + 1
DEPTH_FEATURE_DIM = 64 * 5 * 8
HEIGHT_FEATURE_DIM = 64 * 7 * 7


def _make_robot_cfg(index: int):
    return DJI_FPV_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}")


def _make_camera_cfg(index: int):
    return RayCasterCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}/body",
        mesh_prim_paths=[STATIC_COLLISION_MESH_PRIM_PATH],
        update_period=0.0,
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


def _make_height_scanner_cfg(index: int):
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
        debug_vis=False,
        mesh_prim_paths=[STATIC_COLLISION_MESH_PRIM_PATH],
        update_period=0.0,
    )


def _make_contact_cfg(index: int):
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}/body",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )


@configclass
class DroneSwarmSceneCfg(InteractiveSceneCfg):
    static_mesh = AssetBaseCfg(
        prim_path=STATIC_VISUAL_MESH_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(STATIC_SCAN_DIR, "DR_static_mesh.usdc"),
        ),
    )

    robot_0 = _make_robot_cfg(0)
    robot_1 = _make_robot_cfg(1)
    robot_2 = _make_robot_cfg(2)
    robot_3 = _make_robot_cfg(3)
    robot_4 = _make_robot_cfg(4)

    raycast_camera_0 = _make_camera_cfg(0)
    raycast_camera_1 = _make_camera_cfg(1)
    raycast_camera_2 = _make_camera_cfg(2)
    raycast_camera_3 = _make_camera_cfg(3)
    raycast_camera_4 = _make_camera_cfg(4)

    height_scanner_critic_0 = _make_height_scanner_cfg(0)
    height_scanner_critic_1 = _make_height_scanner_cfg(1)
    height_scanner_critic_2 = _make_height_scanner_cfg(2)
    height_scanner_critic_3 = _make_height_scanner_cfg(3)
    height_scanner_critic_4 = _make_height_scanner_cfg(4)

    contact_forces_0 = _make_contact_cfg(0)
    contact_forces_1 = _make_contact_cfg(1)
    contact_forces_2 = _make_contact_cfg(2)
    contact_forces_3 = _make_contact_cfg(3)
    contact_forces_4 = _make_contact_cfg(4)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class DroneSwarmNavigationEnvCfg(DirectMARLEnvCfg):
    decimation = int((1.0 / 0.005) / PLANNING_FREQ)
    episode_length_s = 60.0
    is_finite_horizon = True

    possible_agents = [f"drone_{idx}" for idx in range(NUM_AGENTS)]
    action_spaces = {agent: 3 for agent in possible_agents}
    observation_spaces = {
        agent: {
            "policy": POLICY_PROPRIO_DIM + DEPTH_FEATURE_DIM,
            "critic": CRITIC_PROPRIO_DIM + HEIGHT_FEATURE_DIM + DEPTH_FEATURE_DIM,
        }
        for agent in possible_agents
    }
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=decimation,
    )
    scene: DroneSwarmSceneCfg = DroneSwarmSceneCfg(
        num_envs=128,
        env_spacing=24.0,
        replicate_physics=False,
    )

    static_visual_mesh_prim_path: str = STATIC_VISUAL_MESH_PRIM_PATH
    map_mesh_prim_path: str = STATIC_COLLISION_MESH_PRIM_PATH
    boundary_walls_root_path: str = "/World/Boundaries"
    boundary_wall_padding: float = 2.0

    surface_bbox_data_path: str = os.path.join(STATIC_SCAN_DIR, "DR_Surface_BBox_Data.txt")
    guidance_trajectories_data_path: str = os.path.join(STATIC_SCAN_DIR, "all_region_pair_trajectories.json")
    precomputed_safe_points_path: str = DEFAULT_PRECOMPUTED_SAFE_POINTS_PATH

    flight_height: float = 1.2
    point_clearance: float = 0.15
    safe_point_grid_spacing: float = 0.25
    guidance_trajectory_eval_dt: float = 0.05
    guidance_arc_length_spacing: float = 0.2

    action_scale: tuple[float, float, float] = (2.5, 2.5, 1.5)
    max_speed: float = 2.5
    teammate_observation_radius: float = 6.0
    disable_teammate_observations: bool = False
    solo_pretraining: bool = False
    independent_agent_goals: bool = False
    teammate_velocity_normalization: float = 5.0
    teammate_distance_penalty_start: float = 0.50
    teammate_distance_penalty_full: float = 0.32
    solo_goal_radius: float = 0.50
    solo_entry_radius: float = 0.75

    initial_formation_offsets_xy: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.8, 0.0),
        (-0.8, 0.0),
        (0.0, 0.8),
        (0.0, -0.8),
    )
    initial_formation_scales: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55)
    cluster_sampling_attempts: int = 32
    spawn_candidate_pool: int = 64
    spawn_assignment_radius: float = 0.50
    min_spawn_separation: float = 0.45
    min_separation: float = 0.65
    collision_distance: float = 0.32
    max_cohesion_radius: float = 1.80
    fallback_cluster_radius: float = 1.80
    fallback_cluster_pairwise_distance: float = 2.00
    max_pairwise_separation: float = 2.20

    goal_soft_sigma: float = 1.0
    goal_tight_sigma: float = 0.2
    cluster_goal_bonus_sigma: float = 3.0
    cluster_entry_radius: float = 1.8
    cluster_success_radius: float = 0.8
    agent_entry_radius: float = 2.4
    agent_goal_radius: float = 1.3
    success_max_centroid_radius: float = 1.05
    success_max_pairwise_distance: float = 1.90
    contact_force_threshold: float = 0.02
    contact_termination_force_threshold: float = 0.08
    contact_termination_steps: int = 3

    guidance_progress_clamp: float = 0.3
    guidance_wrong_way_clamp: float = 0.2
    guidance_lateral_sigma: float = 0.6

    reward_progress_weight: float = 0.7
    reward_wrong_way_weight: float = 0.3
    reward_lateral_weight: float = 0.25
    reward_goal_soft_weight: float = 0.25
    reward_goal_tight_weight: float = 1.5
    reward_cluster_goal_bonus_weight: float = 1.25
    reward_enter_target_region_weight: float = 2.0
    reward_success_weight: float = 4.0
    reward_close_weight: float = 0.8
    reward_far_weight: float = 0.35
    reward_pairwise_far_weight: float = 0.2
    reward_collision_weight: float = 15.0
    reward_contact_weight: float = 6.0
    reward_termination_weight: float = 25.0
    reward_overspeed_weight: float = 0.15
    reward_action_rate_weight: float = 0.05
    reward_teammate_proximity_weight: float = 2.0

    def __post_init__(self):
        self.sim.disable_contact_processing = False
        self.scene.robot_0.spawn.rigid_props.disable_gravity = True
        self.scene.robot_1.spawn.rigid_props.disable_gravity = True
        self.scene.robot_2.spawn.rigid_props.disable_gravity = True
        self.scene.robot_3.spawn.rigid_props.disable_gravity = True
        self.scene.robot_4.spawn.rigid_props.disable_gravity = True

        initialize_depth_noise_generator(
            camera_config=get_camera_config("b2w", use_default_fallback=True),
            use_jit_precompiled=False,
        )

        camera_update_period = self.decimation * self.sim.dt
        self.scene.raycast_camera_0.update_period = camera_update_period
        self.scene.raycast_camera_1.update_period = camera_update_period
        self.scene.raycast_camera_2.update_period = camera_update_period
        self.scene.raycast_camera_3.update_period = camera_update_period
        self.scene.raycast_camera_4.update_period = camera_update_period

        self.scene.height_scanner_critic_0.update_period = camera_update_period
        self.scene.height_scanner_critic_1.update_period = camera_update_period
        self.scene.height_scanner_critic_2.update_period = camera_update_period
        self.scene.height_scanner_critic_3.update_period = camera_update_period
        self.scene.height_scanner_critic_4.update_period = camera_update_period

        self.scene.contact_forces_0.update_period = self.sim.dt
        self.scene.contact_forces_1.update_period = self.sim.dt
        self.scene.contact_forces_2.update_period = self.sim.dt
        self.scene.contact_forces_3.update_period = self.sim.dt
        self.scene.contact_forces_4.update_period = self.sim.dt


@configclass
class DroneSwarmNavigationEnvCfg_DEV(DroneSwarmNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class DroneSwarmNavigationEnvCfg_SOLO(DroneSwarmNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.solo_pretraining = True
        self.independent_agent_goals = True


@configclass
class DroneSwarmNavigationEnvCfg_SOLO_DEV(DroneSwarmNavigationEnvCfg_SOLO):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


@configclass
class DroneSwarmNavigationEnvCfg_PLAY(DroneSwarmNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.sim.render_interval = 4
