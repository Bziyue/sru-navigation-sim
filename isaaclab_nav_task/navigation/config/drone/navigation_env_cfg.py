"""Configuration for planar drone navigation on a static scanned mesh."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab_nav_task.navigation.assets import DJI_FPV_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR
from isaaclab_nav_task.navigation.mdp.custom_noise import DeltaTransformationNoiseCfg
from isaaclab_nav_task.navigation.mdp.delay_manager import ObservationDelayManagerCfg


PLANNING_FREQ = 5.0
STATIC_SCAN_DIR = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Environments", "StaticScan")
STATIC_VISUAL_MESH_PRIM_PATH = "/World/StaticMesh"
STATIC_COLLISION_MESH_PRIM_PATH = "/World/MapMesh"
DEFAULT_PRECOMPUTED_SAFE_POINTS_PATH = os.path.join(
    STATIC_SCAN_DIR,
    "DR_region_safe_points_contact_0p2m_1p2_to_2p0_eroded_0p4m.npz",
)


@configclass
class DroneStaticSceneCfg(InteractiveSceneCfg):
    """Scene with a shared static scan mesh and isolated drones per training environment."""

    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    static_mesh = AssetBaseCfg(
        prim_path=STATIC_VISUAL_MESH_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(STATIC_SCAN_DIR, "DR_static_mesh.usdc"),
        ),
    )

    raycast_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
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

    height_scanner_critic = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
        debug_vis=False,
        mesh_prim_paths=[STATIC_COLLISION_MESH_PRIM_PATH],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
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
class DroneCommandsCfg:
    robot_goal = mdp.StaticRegionGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=False,
        surface_bbox_data_path=os.path.join(STATIC_SCAN_DIR, "DR_Surface_BBox_Data.txt"),
        map_mesh_prim_path=STATIC_COLLISION_MESH_PRIM_PATH,
        spawn_polygon_csv_path=os.path.join(STATIC_SCAN_DIR, "polygon_coords.csv"),
        guidance_paths_data_path=os.path.join(STATIC_SCAN_DIR, "all_region_pair_paths.txt"),
        guidance_trajectories_data_path=os.path.join(STATIC_SCAN_DIR, "all_region_pair_trajectories.json"),
        precomputed_safe_points_path=DEFAULT_PRECOMPUTED_SAFE_POINTS_PATH,
        flight_height=1.2,
        point_clearance=0.15,
        safe_point_grid_spacing=0.25,
        region_points_per_region=192,
        region_center_bias_ratio=0.7,
        visualize_region_safe_points=False,
        region_safe_points_vis_points_per_region=50,
        visualize_region_boxes=False,
        region_box_vis_height=0.12,
        region_vis_z_offset=0.8,
    )


@configclass
class DroneActionsCfg:
    velocity_command = mdp.DroneSE2ActionCfg(
        asset_name="robot",
        scale=[2.5, 2.5, 1.5, 0.15],
        offset=[0.0, 0.0, 0.0, 0.0],
        use_raw_actions=True,
        policy_distr_type="gaussian",
        nominal_height=1.2,
        min_height=0.8,
        max_height=2.5,
        body_name="body",
    )


@configclass
class DroneObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_delayed, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped_delayed,
            params={"command_name": "robot_goal", "flatten": True},
            noise=DeltaTransformationNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False),
        )
        depth_image = ObsTerm(
            func=mdp.depth_image_noisy_delayed,
            params={"sensor_cfg": SceneEntityCfg("raycast_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped,
            params={"command_name": "robot_goal", "flatten": True},
        )
        time_normalized = ObsTerm(func=mdp.time_normalized, params={"command_name": "robot_goal"})
        height_scan_critic = ObsTerm(
            func=mdp.height_scan_feat,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")},
        )
        depth_image = ObsTerm(
            func=mdp.depth_image_prefect,
            params={"sensor_cfg": SceneEntityCfg("raycast_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MetricsCfg(ObsGroup):
        in_goal = ObsTerm(func=mdp.in_goal, params={"flat": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    metrics: MetricsCfg = MetricsCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DroneEventCfg:
    setup_static_scan_world = EventTerm(
        func=mdp.setup_static_scan_world,
        mode="prestartup",
        params={
            "source_prim_expr": STATIC_VISUAL_MESH_PRIM_PATH,
            "output_prim_path": STATIC_COLLISION_MESH_PRIM_PATH,
            "walls_root_path": "/World/Boundaries",
            "wall_padding": 2.0,
            "hide_merged_mesh": True,
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (1.0, 1.0)},
    )

    reset_delay_buffer = EventTerm(func=mdp.reset_and_randomize_delay_buffer, mode="reset")


@configclass
class DroneRewardsCfg:
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.05)
    height_band_violation = RewTerm(
        func=mdp.height_band_violation,
        weight=-10.0,
        params={"min_height": 0.8, "max_height": 2.5},
    )
    height_action_rate_l1 = RewTerm(
        func=mdp.height_action_rate_l1,
        weight=-0.08,
        params={"action_term_name": "velocity_command"},
    )
    height_reference_l2 = RewTerm(
        func=mdp.cruise_height_l2,
        weight=-0.08,
        params={"target_height": 1.2, "release_distance": 1.0, "command_name": "robot_goal", "flat": False},
    )
    guidance_progress = RewTerm(
        func=mdp.guidance_progress_reward,
        weight=0.7,
        params={"command_name": "robot_goal", "clamp_delta": 0.3, "lateral_sigma": 0.6},
    )
    guidance_wrong_way = RewTerm(
        func=mdp.guidance_wrong_way_penalty,
        weight=-0.3,
        params={"command_name": "robot_goal", "clamp_delta": 0.2},
    )
    guidance_lateral_error = RewTerm(
        func=mdp.guidance_lateral_error_penalty,
        weight=-0.25,
        params={"command_name": "robot_goal", "sigma": 0.6},
    )
    episode_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)
    reach_goal_xyz_soft = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.25,
        params={"command_name": "robot_goal", "sigmoid": 2.5, "T_r": 1.0, "probability": 0.01, "flat": False, "ratio": False},
    )
    reach_goal_xyz_tight = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=1.5,
        params={"command_name": "robot_goal", "sigmoid": 0.25, "T_r": 0.1, "probability": 0.01, "flat": False, "ratio": False},
    )


@configclass
class DroneTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out_navigation, time_out=True, params={"distance_threshold": 0.5, "flat": False})
    body_contact = DoneTerm(
        func=mdp.illegal_contact_navigation,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]), "threshold": 0.01},
    )
    early_termination = DoneTerm(
        func=mdp.at_goal_navigation,
        time_out=True,
        params={"distance_threshold": 0.5, "flat": False},
    )
    terrain_fall = DoneTerm(
        func=mdp.terrain_fall,
        time_out=True,
        params={"fall_height_threshold": 0.5},
    )


@configclass
class DroneStaticNavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: DroneStaticSceneCfg = DroneStaticSceneCfg(num_envs=512, env_spacing=24.0, replicate_physics=False)
    observations: DroneObservationsCfg = DroneObservationsCfg()
    actions: DroneActionsCfg = DroneActionsCfg()
    commands: DroneCommandsCfg = DroneCommandsCfg()
    rewards: DroneRewardsCfg = DroneRewardsCfg()
    terminations: DroneTerminationsCfg = DroneTerminationsCfg()
    events: DroneEventCfg = DroneEventCfg()
    delay_cfg: ObservationDelayManagerCfg = ObservationDelayManagerCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.is_finite_horizon = True
        self.decimation = int((1 / self.sim.dt) / PLANNING_FREQ)
        self.episode_length_s = 60.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False

        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config
        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        initialize_depth_noise_generator(camera_config=get_camera_config("b2w", use_default_fallback=True), use_jit_precompiled=False)

        if self.scene.height_scanner_critic is not None:
            self.scene.height_scanner_critic.update_period = self.decimation * self.sim.dt
        if self.scene.raycast_camera is not None:
            self.scene.raycast_camera.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class DroneStaticNavigationEnvCfg_DEV(DroneStaticNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class DroneStaticNavigationEnvCfg_PLAY(DroneStaticNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.robot_goal.debug_vis = True
        self.observations.policy.enable_corruption = False


@configclass
class DroneStaticNavigationEnvCfg_PLAY_FAST(DroneStaticNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.robot_goal.debug_vis = False
        self.commands.robot_goal.visualize_region_safe_points = False
        self.commands.robot_goal.visualize_region_boxes = False
        self.observations.policy.enable_corruption = False
        # Render more frequently than the 5 Hz control loop so local playback looks smooth.
        self.sim.render_interval = 4
