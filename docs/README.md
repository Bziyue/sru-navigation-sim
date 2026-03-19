# IsaacLab Navigation Extension - SRU Project

[![Paper](https://img.shields.io/badge/IJRR-2025-blue)](https://journals.sagepub.com/home/ijr)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://michaelfyang.github.io/sru-project-website/)

> **📌 Important Note**: This repository contains the **IsaacLab task extension** for the SRU project, providing diverse navigation environments with dynamic obstacle configurations and terrain variations. This repository does **not** include the `rsl_rl` learning module (network architectures, PPO/MDPO training algorithms). See the [project website](https://michaelfyang.github.io/sru-project-website/) for the complete navigation system.

## Overview

A standalone, self-contained IsaacLab task extension for visual navigation in Isaac Lab v2.1.1 (Isaac Sim 4.5). This repository provides:

- **Environment**: Diverse navigation environments in IsaacLab with dynamic obstacle configurations and terrain variations
- **Task Definition**: Hierarchical control architecture interface for visual navigation with reinforcement learning
- **Simulation**: High-fidelity physics simulation with realistic depth sensor noise

**Note**: This repository focuses on the simulation environment and task definition. The RL training infrastructure (neural network architectures, PPO/MDPO algorithms) is provided by the separate `rsl_rl` learning module.

This extension implements a hierarchical control architecture for visual navigation:
- **High-level policy**: Learns to output SE2 velocity commands (vx, vy, omega) at 5Hz
- **Low-level policy**: Pre-trained locomotion policy that converts velocity commands to joint actions at 50Hz

The extension is fully self-contained with all necessary robot models, materials, and pre-trained locomotion policies included.

### What's Included

- ✅ IsaacLab task extension for visual navigation environments
- ✅ Maze terrain generation with curriculum learning
- ✅ Self-contained assets: Robot models (USD), locomotion policies, depth encoders
- ✅ Multiple robot platforms: B2W (bipedal wheeled) and AoW-D (Anymal on Wheels)
- ✅ Observation definitions: Depth images, proprioception, goal commands
- ✅ Reward functions: Goal reaching, action smoothing, movement penalties
- ✅ Hierarchical action interface: SE2 velocity commands to low-level controllers
- ✅ Domain randomization: Camera pose, action scaling, low-pass filters, sensor delays
- ✅ Training scripts compatible with RSL-RL (PPO/MDPO algorithms)

### What's NOT Included

- ❌ `rsl_rl` learning module (network architectures, PPO/MDPO training algorithms)
- ❌ Neural network structures for high-level navigation policy
- ❌ On-policy RL training algorithms (PPO/MDPO implementations)

**Note**: The `rsl_rl` package must be installed separately to train navigation policies. See the Installation section below.

### Related Projects

- [sru-pytorch-spatial-learning](https://github.com/michaelfyang/sru-pytorch-spatial-learning) - Core SRU architecture
- [SRU Project Website](https://michaelfyang.github.io/sru-project-website/) - Complete navigation system

## Features

- **Visual navigation** using depth cameras with realistic noise simulation
- **Maze terrain generation** with curriculum learning
- **Self-contained assets**: All robot models and locomotion policies included
- **Multiple robot platforms**:
  - **B2W**: Bipedal wheeled robot (with ZedX camera)
  - **AoW-D**: Anymal on Wheels (with ZedX camera)
- **Asymmetric actor-critic** with privileged critic observations
- **Curriculum learning** for terrain difficulty progression
- **Multiple algorithms**: MDPO and PPO support via RSL-RL
- **Domain randomization**: Camera pose, action scaling, low-pass filters, sensor delays

## Installation

### Prerequisites

- Isaac Lab v2.1.1 installed and configured
- Isaac Sim 4.5.0
- Python 3.10
- PyTorch >= 2.5.1

### Step 1: Clone or Place the Extension

This extension should be placed in the `source/` directory of your IsaacLab installation:

```bash
# Navigate to your IsaacLab installation
cd /path/to/IsaacLab

# If cloning this repository separately, place it in source/
# Your directory structure should look like:
# IsaacLab/
# ├── source/
# │   ├── isaaclab/
# │   ├── isaaclab_assets/
# │   └── isaaclab_nav_task/  <- This extension
```

### Step 2: Install the Extension

Install the extension in development mode from the IsaacLab root directory:

```bash
# From IsaacLab root directory
./isaaclab.sh -p -m pip install -e source/isaaclab_nav_task

# Or navigate to the extension directory
cd source/isaaclab_nav_task
../../isaaclab.sh -p -m pip install -e .
```

### Step 3: Install RSL-RL (Required for Training)

This extension requires the `rsl_rl` package for training. Install the custom version with MDPO/PPO algorithms:

```bash
# Clone and install custom rsl_rl (if not already installed)
cd /path/to/your/workspace
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
pip install -e .
```

### Verify Installation

Test that the extension is properly installed:

```bash
# From IsaacLab root directory
./isaaclab.sh -p -m pip show isaaclab_nav_task

# List available tasks
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py --help
```

You should see the task IDs listed (e.g., `Isaac-Nav-MDPO-B2W-v0`, `Isaac-Nav-PPO-AoW-D-v0`, etc.).

## Available Tasks

### B2W
| Task ID | Description |
|---------|-------------|
| `Isaac-Nav-MDPO-B2W-v0` | MDPO training |
| `Isaac-Nav-PPO-B2W-v0` | PPO training |
| `Isaac-Nav-MDPO-B2W-Play-v0` | MDPO playback |
| `Isaac-Nav-PPO-B2W-Play-v0` | PPO playback |
| `Isaac-Nav-MDPO-B2W-Dev-v0` | MDPO development |
| `Isaac-Nav-PPO-B2W-Dev-v0` | PPO development |

### AoW-D
| Task ID | Description |
|---------|-------------|
| `Isaac-Nav-MDPO-AoW-D-v0` | MDPO training |
| `Isaac-Nav-PPO-AoW-D-v0` | PPO training |
| `Isaac-Nav-MDPO-AoW-D-Play-v0` | MDPO playback |
| `Isaac-Nav-PPO-AoW-D-Play-v0` | PPO playback |
| `Isaac-Nav-MDPO-AoW-D-Dev-v0` | MDPO development |
| `Isaac-Nav-PPO-AoW-D-Dev-v0` | PPO development |

## Training

### Using the standalone training script

```bash
# Train B2W with PPO
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 4096 --headless

# Train AoW-D with PPO
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-AoW-D-v0 --num_envs 4096 --headless

# Train with custom wandb run name
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096 --headless \
    --run_name "experiment_v1_with_curriculum"

# Train with multiple custom parameters
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 2048 --headless \
    --run_name "large_training_run" --seed 42 --max_iterations 20000
```

### Development/Testing (smaller config with tensorboard)

The `-Dev-v0` variants use tensorboard logging instead of wandb and have reduced iterations (300 vs 15000) for quick testing:

```bash
# Quick test with small environment count
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-Dev-v0 --num_envs 32 --headless
```

### Using the standard RSL-RL workflow

```bash
# Train with RSL-RL
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096
```

## Playing Trained Policies

```bash
# Play using standalone script
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 --num_envs 16

# Play with specific checkpoint
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 \
    --checkpoint /path/to/model.pt
```

## Architecture

```
isaaclab_nav_task/
├── config/
│   └── extension.toml            # Extension metadata
├── docs/
│   └── README.md                 # This file
├── scripts/
│   ├── train.py                  # Training script
│   └── play.py                   # Playback script
├── setup.py                      # Installation script
├── pyproject.toml                # Build configuration
└── isaaclab_nav_task/
    ├── __init__.py               # Extension entry point
    └── navigation/
        ├── __init__.py
        ├── navigation_env_cfg.py         # Base environment config
        ├── assets/                       # Robot configurations and data
        │   ├── __init__.py
        │   ├── b2w.py                    # B2W robot config
        │   ├── aow_d.py                  # AoW-D robot config
        │   └── data/                     # Self-contained asset directory
        │       ├── Robots/               # Robot USD models and materials
        │       │   └── AoW-D/            # AoW-D robot assets
        │       │       ├── aow_d.usd     # Robot USD model
        │       │       └── Props/        # Materials and textures
        │       └── Policies/             # Pre-trained models
        │           ├── depth_encoder/    # VAE depth encoders
        │           │   └── vae_pretrain_new.pth  (ZedX)
        │           └── locomotion/       # Low-level locomotion policies
        │               ├── aow_d/        # policy_blind_3_1.pt (1.7 MB)
        │               └── b2w/          # policy_b2w_new_2.pt (2.0 MB)
        ├── config/
        │   ├── rl_cfg.py                 # Base RL configurations
        │   ├── b2w/
        │   │   ├── __init__.py           # Task registration
        │   │   ├── navigation_env_cfg.py
        │   │   └── agents/
        │   │       └── rsl_rl_cfg.py
        │   └── aow_d/
        │       ├── __init__.py
        │       ├── navigation_env_cfg.py
        │       └── agents/
        │           └── rsl_rl_cfg.py
        ├── mdp/
        │   ├── observations.py       # Observation functions (13 functions)
        │   ├── rewards.py            # Reward functions (5 functions)
        │   ├── terminations.py       # Termination conditions (4 functions)
        │   ├── curriculums.py        # Curriculum terms (1 function)
        │   ├── events.py             # Domain randomization events (5 functions)
        │   ├── depth_utils/          # Depth processing utilities
        │   │   ├── __init__.py
        │   │   ├── camera_config.py      # Camera configurations (ZedX)
        │   │   └── depth_noise_encoder.py # VAE-based depth encoder
        │   └── navigation/
        │       ├── goal_commands.py
        │       ├── goal_commands_cfg.py
        │       └── actions/
        │           ├── __init__.py
        │           ├── navigation_se2_actions.py
        │           └── navigation_se2_actions_cfg.py
        └── terrains/                # Custom terrain generators
            ├── __init__.py
            ├── hf_terrains_maze.py      # Maze terrain generation
            ├── hf_terrains_maze_cfg.py  # Maze terrain configs
            ├── maze_config.py           # Maze parameters
            └── patches.py               # TerrainImporter patches
```

## Compatibility

- **Isaac Lab**: v2.1.1
- **Isaac Sim**: 4.5.0
- **Python**: 3.10
- **PyTorch**: >= 2.5.1

## Self-Contained Assets

The extension includes all necessary assets and does not depend on external asset repositories:

### Robot Models (`assets/data/Robots/`)
- **AoW-D**: Complete USD model with materials and textures
  - Used when AoW-D robots are not available in the base `isaaclab_assets`
  - Includes all necessary Props and material textures (11 baked textures)

### Pre-trained Policies (`assets/data/Policies/`)

**Depth Encoders** (`depth_encoder/`):
- `vae_pretrain_new.pth`: ZedX camera encoder for B2W and AoW-D
- VAE architecture with RegNet backbone + Feature Pyramid Network

**Locomotion Policies** (`locomotion/`):
- `aow_d/policy_blind_3_1.pt` (1.7 MB): AoW-D wheeled locomotion
- `b2w/policy_b2w_new_2.pt` (2.0 MB): B2W bipedal wheeled locomotion

All locomotion policies are pre-trained and loaded by the hierarchical action controller.

## Key Components

### Navigation Environment (`navigation_env_cfg.py`)
- Defines the scene with terrain, robot, and sensors
- Configures observation groups for policy and critic
- Sets up reward terms for goal reaching and movement penalties
- Configures curriculum for terrain difficulty

### MDP Components (`mdp/`)

**Cleaned and optimized** - removed unused functions to improve maintainability:

- **observations.py** (13 functions): Depth image processing, proprioception, goal direction, delay buffers
- **rewards.py** (5 functions): Goal reaching, action smoothing, movement penalties
- **terminations.py** (4 functions): Timeout, collision detection, angle limits, goal reaching
- **curriculums.py** (1 function): Backward movement penalty scheduling
- **events.py** (5 functions): Camera randomization, action scaling, delay buffer management

### Navigation Actions (`mdp/navigation/actions/`)
- Hierarchical action space with SE2 velocity commands
- Integration with pre-trained low-level locomotion policies

### Terrain Generation and Goal Sampling (`terrains/`)

The extension includes custom maze terrain generators built on Isaac Lab's terrain generation system, providing diverse navigation environments with safe goal and spawn position sampling.

**Key Features:**
- **Four terrain types**: Maze, non-maze/random, stairs, and pits
- **Curriculum learning**: 180 terrains organized in 6 difficulty levels
- **Safe position sampling**: Separate padding for goals (0.5m) and spawns (0.6m)
- **Mesh optimization**: ~80-99% vertex reduction for large-scale training
- **Explicit boolean masks**: Pre-computed valid positions for efficient sampling

**Terrain Configuration:**
- Grid: 6 rows (difficulty) × 30 columns (variations) = 180 terrains
- Size: 30m × 30m per terrain with 0.1m resolution (300×300 cells)
- Proportions: 30% maze, 20% random, 30% stairs, 20% pits

For detailed documentation on terrain generation, goal/spawn sampling, coordinate systems, and implementation details, see [TERRAIN_AND_GOALS.md](TERRAIN_AND_GOALS.md).

### Depth Processing (`mdp/depth_utils/`)
- **DepthNoise**: Simulates realistic stereo camera noise using disparity-based filtering
- **DepthNoiseEncoder**: VAE-based depth encoder using RegNet backbone with Feature Pyramid Network
- **Camera Configurations**: Pre-defined configs for different camera types:

| Camera | Robots | Resolution | Depth Range | Encoder |
|--------|--------|------------|-------------|---------|
| ZedX | B2W, AoW-D | 64x40 | 0.25-10.0m | `vae_pretrain_new.pth` |

### Custom Robot Assets (`assets/`)

Robot configuration modules define robot-specific parameters:

**B2W** (`b2w.py`):
- Actuator configurations (position/velocity control)
- Initial joint states
- USD asset path (from base `isaaclab_assets`)

**AoW-D** (`aow_d.py`):
- Actuator configurations for wheeled quadruped
- Initial joint states
- USD asset path (from local `assets/data/Robots/AoW-D/`)
- Uses local robot model when not available in base assets

Both configurations integrate seamlessly with the hierarchical navigation controller and pre-trained locomotion policies.

## Cluster Deployment

This repository does **not** include the old `docker/cluster` helper scripts referenced in earlier internal workflows. For scheduler-based execution, wrap the training entrypoint in your own Slurm or container submission script.

Use the entrypoint that exists in this repository:

```bash
python sru-navigation-sim/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 \
    --num_envs 2048 \
    --max_iterations 10000 \
    --headless \
    --enable_cameras \
    --device cuda:0
```

Guidance:

- The correct training entrypoint is `sru-navigation-sim/scripts/train.py`.
- Navigation tasks in this repository should launch with `--enable_cameras`.
- On multi-GPU or cluster jobs, pass the scheduler-assigned device explicitly with `--device cuda:N`.

If you need cluster/container deployment, follow the upstream [IsaacLab cluster guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html#cluster-guide) and adapt it to the repository paths above.

### Troubleshooting

**Git ownership errors**: Rebuild Docker image (includes fix) or run in container:
```bash
git config --global --add safe.directory '*'
```

**Memory issues**: Reduce `--num_envs` or increase `#SBATCH --mem-per-cpu`

## License

MIT License - See [LICENSE](../LICENSE) file for details

Copyright (c) 2025 Fan Yang, Per Frivik, Robotic Systems Lab, ETH Zurich

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

## Contact

**Authors**:
- Fan Yang (fanyang1@ethz.ch)
- Per Frivik (pfrivik@ethz.ch)

**Affiliation**: Robotic Systems Lab, ETH Zurich
