import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Nav-IPPO-Drone-Swarm-Static-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_nav_task.navigation.config.drone_swarm.swarm_env_cfg:DroneSwarmNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmNavIPPORunnerCfg,
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nav-IPPO-Drone-Swarm-Static-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_nav_task.navigation.config.drone_swarm.swarm_env_cfg:DroneSwarmNavigationEnvCfg_DEV",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmNavIPPORunnerDevCfg,
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Nav-IPPO-Drone-Swarm-Static-Solo-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_nav_task.navigation.config.drone_swarm.swarm_env_cfg:DroneSwarmNavigationEnvCfg_SOLO",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmNavIPPOSoloRunnerCfg,
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nav-IPPO-Drone-Swarm-Static-Solo-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_nav_task.navigation.config.drone_swarm.swarm_env_cfg:DroneSwarmNavigationEnvCfg_SOLO_DEV",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmNavIPPOSoloRunnerDevCfg,
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nav-IPPO-Drone-Swarm-Static-Play-v0",
    entry_point="isaaclab_nav_task.navigation.swarm_navigation_env:DroneSwarmNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_nav_task.navigation.config.drone_swarm.swarm_env_cfg:DroneSwarmNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DroneSwarmNavIPPORunnerCfg,
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)
