# MACE-RL: Meta-Adaptive Curiosity-Driven Exploration with Episodic Memory

MACE-RL is a novel reinforcement learning framework that learns to dynamically regulate its own exploration strategy. It integrates:
- A **curiosity module** informed by episodic memory
- A **meta-learning network** that adaptively regulates the curiosity bonus

This enables agents to balance exploration and exploitation dynamically.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test
```bash
python experiment_runner.py --quick-test
```

### Run Experiments
```bash
python experiment_runner.py --run-experiments
```

### Publication Mode
```bash
python experiment_runner.py --publication-mode
```

### Visualize Agent
```bash
python visualize_agent.py --model-path path/to/model.pth --env CartPole-v1
```

## Project Structure

```
mace_rl/
├── agents/         # RL agents (PPO)
├── modules/        # Core MACE-RL components
│   ├── curiosity.py        # Curiosity module
│   ├── episodic_memory.py  # Episodic memory
│   └── meta_adaptation.py  # Meta-learning network
├── envs/           # Environment wrappers
└── utils/          # Utilities and helpers
```

## Results

| Algorithm | Acrobot-v1 | CartPole-v1 | HopperBulletEnv-v0 |
|-----------|------------|-------------|-------------------|
| PPO (Baseline) | 186 / 617 | 140 / 660 | 865 / 1841 |
| MACE-RL (Fixed β) | 201 / 595 | 120 / 620 | 782 / 1801 |
| **MACE-RL (Full)** | **249 / 586** | **104 / 634** | **495 / 1799** |

*Metrics: Sample Efficiency / Convergence Steps (lower is better)*