#!/usr/bin/env python3
"""
MACE-RL Experiment Runner
=========================

A streamlined experiment runner for MACE-RL research.

Usage:
    python experiment_runner.py --quick-test              # Quick test mode
    python experiment_runner.py --run-experiments         # Run all experiments
    python experiment_runner.py --run-ablation            # Run ablation studies
    python experiment_runner.py --publication-mode        # Publication-quality experiments
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module="gym")

import argparse
import os
import json
import pickle
import logging
import numpy as np
import torch
import gym
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from scipy import stats
from tqdm import tqdm

from mace_rl.utils.utils import set_seed
from mace_rl.utils.logger import setup_logger
from mace_rl.agents.ppo import PPO
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.utils.reward_system import HybridRewardSystem
from mace_rl.utils.beta_schedulers import create_beta_scheduler
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env

# Experiment configuration
EXPERIMENT_CONFIG = {
    'environments': [
        'CartPole-v1',
        'Acrobot-v1',
        'HopperBulletEnv-v0'
    ],
    'algorithms': ['mace_rl_full'],
    'seeds': [42],
    'max_episodes': 6000,
    'max_timesteps': 10000,
    'env_specific_episodes': {
        'CartPole-v1': 2000,
        'Acrobot-v1': 2000,
        'HopperBulletEnv-v0': 6000,
    },
    'env_specific_timesteps': {
        'CartPole-v1': 500,
        'Acrobot-v1': 500,
        'HopperBulletEnv-v0': 1000,
    },
    'use_meta_network_for_beta': True,
    'ablation_studies': {
        'beta_values': [0.75, 1.0, 1.5],
        'beta_schedulers': ['constant', 'linear_decay', 'exponential_decay']
    },
    'publication_mode': {
        'seeds': [42],
        'max_episodes': 2000,
        'max_timesteps': 20000,
    }
}


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_nested_dict():
    """Helper function for defaultdict to avoid pickle issues."""
    return defaultdict(list)


class ExperimentRunner:
    """Main class for running MACE-RL experiments."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(results_dir, f'mace_rl_suite_{self.timestamp}')
        
        # Create directory structure
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
        
        self.logger = setup_logger(
            'experiment_suite',
            os.path.join(self.experiment_dir, 'logs', 'suite.log'),
            level=logging.INFO,
            console_output=True
        )
        
        self.results = defaultdict(create_nested_dict)
        self._validate_environments()

    def _validate_environments(self):
        """Validate that configured environments can be loaded."""
        self.logger.info("Validating environments...")
        for env_name in EXPERIMENT_CONFIG['environments']:
            try:
                env = self._create_environment(env_name)
                env.reset()
                env.close()
                self.logger.info(f"✅ {env_name} validated")
            except Exception as e:
                self.logger.error(f"❌ {env_name} failed: {e}")

    def _create_environment(self, env_name):
        """Create and return an environment."""
        if "MiniGrid" in env_name:
            return make_minigrid_env(env_name)
        elif "BulletEnv" in env_name:
            return make_pybullet_env(env_name)
        else:
            return gym.make(env_name)

    def _initialize_components(self, algorithm, state_dim, action_dim, 
                               has_continuous_action_space, is_atari, ablation_config=None):
        """Initialize algorithm components."""
        # Extract configuration
        memory_size = ablation_config.get('memory_size', 1000) if ablation_config else 1000
        beta_initial = ablation_config.get('beta_initial', 0.4) if ablation_config else 0.4
        beta_scheduler_config = ablation_config.get('beta_scheduler') if ablation_config else None
        
        # Initialize PPO agent
        agent = PPO(
            state_dim, action_dim,
            lr_actor=0.0003, lr_critic=0.001,
            gamma=0.99, K_epochs=4, eps_clip=0.2,
            has_continuous_action_space=has_continuous_action_space
        )
        
        components = {'agent': agent, 'episodic_memory': None, 
                     'curiosity_module': None, 'reward_system': None}
        
        if algorithm == 'ppo_baseline':
            return components
        
        # MACE-RL components
        if not is_atari and algorithm.startswith('mace_rl'):
            state_size = state_dim if isinstance(state_dim, int) else state_dim[0]
            
            # Episodic memory
            if 'no_memory' not in algorithm:
                components['episodic_memory'] = EpisodicMemory(memory_size, state_size)
            
            # Curiosity module
            if 'no_curiosity' not in algorithm:
                components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, components['episodic_memory'],
                    continuous=has_continuous_action_space
                )
            
            # Reward system
            if beta_scheduler_config:
                components['reward_system'] = HybridRewardSystem(
                    components['curiosity_module'],
                    beta_scheduler=beta_scheduler_config
                )
            else:
                components['reward_system'] = HybridRewardSystem(
                    components['curiosity_module'],
                    beta_initial=beta_initial
                )
        
        return components

    def run_single_experiment(self, env_name, algorithm, seed, episode_limit=None, ablation_config=None):
        """Run a single experiment."""
        # Create experiment ID
        exp_id = f"{env_name}_{algorithm}_seed{seed}"
        if ablation_config:
            config_str = "_".join([f"{k}{v}" for k, v in sorted(ablation_config.items()) 
                                   if not isinstance(v, dict)])
            exp_id += f"_{config_str}"
        exp_id = exp_id.replace(".", "p").replace("-", "n")[:200]
        
        self.logger.info(f"Starting: {exp_id}")
        set_seed(seed)
        
        # Initialize environment
        try:
            env = self._create_environment(env_name)
            has_continuous_action_space = hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0
            is_atari = "NoFrameskip" in env_name
        except Exception as e:
            self.logger.error(f"Failed to create environment {env_name}: {e}")
            return self._create_error_result(env_name, algorithm, seed, exp_id, str(e))
        
        # Get dimensions
        state_dim = env.observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n
        
        # Initialize components
        components = self._initialize_components(
            algorithm, state_dim, action_dim, has_continuous_action_space, is_atari, ablation_config
        )
        
        # Meta-network for beta adaptation
        meta_net = None
        meta_beta_history = []
        use_meta_net = EXPERIMENT_CONFIG.get('use_meta_network_for_beta', False)
        if algorithm == 'mace_rl_full' and not is_atari and use_meta_net:
            from mace_rl.modules.meta_adaptation import MetaAdaptation
            meta_input_dim = (state_dim if isinstance(state_dim, int) else state_dim[0]) + action_dim + 2
            meta_net = MetaAdaptation(meta_input_dim, 128, 1)
            self.logger.info(f"[MetaNet] Using meta-network for beta")
        
        # Training loop
        episode_rewards = []
        curiosity_rewards = []
        total_steps = 0
        
        max_eps = episode_limit or EXPERIMENT_CONFIG['env_specific_episodes'].get(
            env_name, EXPERIMENT_CONFIG['max_episodes'])
        max_timesteps = EXPERIMENT_CONFIG['env_specific_timesteps'].get(
            env_name, EXPERIMENT_CONFIG['max_timesteps'])
        
        pbar = tqdm(range(max_eps), desc=f"Training {algorithm}", unit="episode", ncols=100)
        
        for episode in pbar:
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            
            done = False
            episode_reward = 0
            episode_curiosity = 0
            step_count = 0
            meta_input_seq = []
            
            for t in range(max_timesteps):
                step_count += 1
                total_steps += 1
                action = components['agent'].select_action(state)
                
                next_state, extrinsic_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward = extrinsic_reward
                curiosity_bonus = 0
                
                # Meta-network input collection
                if meta_net is not None:
                    state_tensor = torch.FloatTensor(state).view(-1)
                    if has_continuous_action_space:
                        action_tensor = torch.FloatTensor(action)
                    else:
                        action_tensor = torch.zeros(action_dim)
                        action_tensor[int(action)] = 1.0
                    meta_input = torch.cat([
                        state_tensor, action_tensor,
                        torch.tensor([extrinsic_reward, curiosity_bonus], dtype=torch.float32)
                    ])
                    meta_input_seq.append(meta_input)
                
                # Curiosity reward calculation
                if components['curiosity_module'] and components['reward_system']:
                    try:
                        state_tensor = torch.FloatTensor(state).view(-1)
                        action_tensor = (torch.FloatTensor(action) if has_continuous_action_space 
                                        else torch.LongTensor([action]))
                        total_reward = components['reward_system'].get_total_reward(
                            state_tensor, action_tensor, extrinsic_reward
                        )
                        curiosity_bonus = total_reward - extrinsic_reward
                    except Exception:
                        total_reward = extrinsic_reward
                
                components['agent'].buffer.rewards.append(total_reward)
                components['agent'].buffer.is_terminals.append(done)
                
                # Update episodic memory
                if components['episodic_memory']:
                    try:
                        state_flat = torch.FloatTensor(state).view(-1)
                        next_state_flat = torch.FloatTensor(next_state).view(-1)
                        value = torch.cat([next_state_flat, torch.FloatTensor([total_reward])])
                        components['episodic_memory'].add(state_flat, value)
                    except Exception:
                        pass
                
                state = next_state
                episode_reward += extrinsic_reward
                episode_curiosity += curiosity_bonus
                
                if done:
                    break
            
            # Meta-network update
            if meta_net is not None and meta_input_seq:
                meta_seq_tensor = torch.stack(meta_input_seq).unsqueeze(0)
                meta_net.train()
                meta_beta_out, _ = meta_net(meta_seq_tensor)
                meta_beta_value = float(torch.sigmoid(meta_beta_out[0, 0]).item())
                meta_beta_history.append(meta_beta_value)
                if components['reward_system']:
                    components['reward_system'].beta = meta_beta_value
                meta_loss = torch.tensor(-episode_reward, dtype=torch.float32, requires_grad=True)
                meta_net.update_meta(meta_loss)
            
            # Policy update
            try:
                components['agent'].update()
            except Exception:
                pass
            
            # Beta scheduler update
            if meta_net is None and episode % 10 == 0 and components['reward_system']:
                components['reward_system'].update_beta(episode=episode, step=total_steps)
            
            episode_rewards.append(episode_reward)
            curiosity_rewards.append(episode_curiosity)
            
            if len(episode_rewards) >= 10:
                pbar.set_postfix({"Reward": f"{np.mean(episode_rewards[-10:]):.2f}"})
        
        pbar.close()
        
        # Save model
        model_path = os.path.join(self.experiment_dir, 'models', f"{exp_id}_final.pth")
        components['agent'].save(model_path)
        
        # Create result
        result = {
            'episode_rewards': episode_rewards,
            'curiosity_rewards': curiosity_rewards,
            'meta_beta_history': meta_beta_history if meta_net else None,
            'final_reward': episode_rewards[-1] if episode_rewards else 0,
            'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0,
            'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0,
            'config': {
                'env_name': env_name, 'algorithm': algorithm, 'seed': seed,
                'max_episodes': max_eps, 'ablation_config': ablation_config, 'exp_id': exp_id
            }
        }
        
        # Save result
        data_path = os.path.join(self.experiment_dir, 'data', f"{exp_id}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(result, f)
        
        self.logger.info(f"✅ Completed: {exp_id} - Mean: {result['mean_reward']:.2f}")
        return result

    def _create_error_result(self, env_name, algorithm, seed, exp_id, error):
        """Create error result dict."""
        return {
            'config': {'env_name': env_name, 'algorithm': algorithm, 'seed': seed, 'exp_id': exp_id},
            'error': error, 'mean_reward': 0.0, 'std_reward': 0.0,
            'final_reward': 0.0, 'episode_rewards': []
        }

    def run_all_experiments(self, parallel=False, beta_override=None):
        """Run all experiment configurations."""
        tasks = []
        for env_name in EXPERIMENT_CONFIG['environments']:
            for algorithm in EXPERIMENT_CONFIG['algorithms']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    ablation_config = {'beta_initial': beta_override} if beta_override else None
                    tasks.append((env_name, algorithm, seed, None, ablation_config))
        
        print(f"\n{'='*60}")
        print(f"Running {len(tasks)} experiments")
        print(f"{'='*60}\n")
        
        results = []
        if parallel and len(tasks) > 1:
            with Pool(processes=min(cpu_count(), 4)) as pool:
                results = pool.starmap(self.run_single_experiment, tasks)
        else:
            for task in tasks:
                results.append(self.run_single_experiment(*task))
        
        for result in results:
            env_name = result['config']['env_name']
            algorithm = result['config']['algorithm']
            self.results[env_name][algorithm].append(result)
        
        return results

    def run_ablation_studies(self, ablation_type='all', parallel=False):
        """Run ablation studies."""
        self.logger.info(f"Starting ablation studies: {ablation_type}")
        
        tasks = []
        base_env = 'HopperBulletEnv-v0'
        base_algorithm = 'mace_rl_full'
        
        if ablation_type in ['all', 'beta_values']:
            for beta in EXPERIMENT_CONFIG['ablation_studies']['beta_values']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    tasks.append((base_env, base_algorithm, seed, None, {'beta_initial': beta}))
        
        if ablation_type in ['all', 'beta_schedulers']:
            for scheduler in EXPERIMENT_CONFIG['ablation_studies']['beta_schedulers']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    tasks.append((base_env, base_algorithm, seed, None, {'beta_scheduler': scheduler}))
        
        results = []
        if parallel and len(tasks) > 1:
            with Pool(processes=min(cpu_count(), 4)) as pool:
                results = pool.starmap(self.run_single_experiment, tasks)
        else:
            for task in tasks:
                results.append(self.run_single_experiment(*task))
        
        return results

    def run_publication_experiments(self, parallel=False, ablation=False):
        """Run publication-quality experiments."""
        self.logger.info("Starting publication experiments")
        
        # Save original config
        orig_config = {
            'seeds': EXPERIMENT_CONFIG['seeds'],
            'max_episodes': EXPERIMENT_CONFIG['max_episodes'],
            'max_timesteps': EXPERIMENT_CONFIG['max_timesteps']
        }
        
        # Apply publication settings
        pub_config = EXPERIMENT_CONFIG['publication_mode']
        EXPERIMENT_CONFIG['seeds'] = pub_config['seeds']
        EXPERIMENT_CONFIG['max_episodes'] = pub_config['max_episodes']
        EXPERIMENT_CONFIG['max_timesteps'] = pub_config['max_timesteps']
        
        try:
            results = self.run_all_experiments(parallel=parallel)
            if ablation:
                self.run_ablation_studies(parallel=parallel)
        finally:
            # Restore original config
            EXPERIMENT_CONFIG['seeds'] = orig_config['seeds']
            EXPERIMENT_CONFIG['max_episodes'] = orig_config['max_episodes']
            EXPERIMENT_CONFIG['max_timesteps'] = orig_config['max_timesteps']
        
        return results

    def analyze_results(self):
        """Analyze experimental results."""
        self.logger.info("Analyzing results")
        
        if not self.results:
            self._load_results()
        
        if not self.results:
            self.logger.warning("No results to analyze")
            return
        
        stats_results = self._statistical_analysis()
        self._generate_report(stats_results)
        self._export_results()

    def _load_results(self):
        """Load results from data files."""
        data_dir = os.path.join(self.experiment_dir, 'data')
        if not os.path.exists(data_dir):
            return
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.pkl'):
                with open(os.path.join(data_dir, filename), 'rb') as f:
                    result = pickle.load(f)
                env_name = result['config']['env_name']
                algorithm = result['config']['algorithm']
                self.results[env_name][algorithm].append(result)

    def _statistical_analysis(self):
        """Perform statistical analysis."""
        stats_results = {}
        
        for env_name in self.results:
            stats_results[env_name] = {}
            algorithm_data = {}
            
            for algorithm in self.results[env_name]:
                rewards = [r['mean_reward'] for r in self.results[env_name][algorithm]]
                algorithm_data[algorithm] = rewards
                
                if rewards:
                    stats_results[env_name][algorithm] = {
                        'mean': float(np.mean(rewards)),
                        'std': float(np.std(rewards)),
                        'min': float(np.min(rewards)),
                        'max': float(np.max(rewards)),
                        'n_runs': len(rewards)
                    }
            
            # Pairwise t-tests
            algorithms = list(algorithm_data.keys())
            p_values = {}
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    alg1, alg2 = algorithms[i], algorithms[j]
                    if algorithm_data[alg1] and algorithm_data[alg2]:
                        _, p_val = stats.ttest_ind(algorithm_data[alg1], algorithm_data[alg2])
                        p_values[f"{alg1}_vs_{alg2}"] = float(p_val)
            stats_results[env_name]['p_values'] = p_values
        
        # Save
        stats_path = os.path.join(self.experiment_dir, 'statistical_analysis.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2, default=convert_numpy)
        
        return stats_results

    def _generate_report(self, stats_results):
        """Generate summary report."""
        report_path = os.path.join(self.experiment_dir, 'experiment_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# MACE-RL Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for env_name in stats_results:
                f.write(f"## {env_name}\n\n")
                f.write("| Algorithm | Mean | Std | Min | Max | Runs |\n")
                f.write("|-----------|------|-----|-----|-----|------|\n")
                
                for algorithm in stats_results[env_name]:
                    if algorithm != 'p_values':
                        s = stats_results[env_name][algorithm]
                        f.write(f"| {algorithm} | {s['mean']:.2f} | {s['std']:.2f} | "
                               f"{s['min']:.2f} | {s['max']:.2f} | {s['n_runs']} |\n")
                f.write("\n")

    def _export_results(self):
        """Export results to JSON."""
        summary = []
        for env_name in self.results:
            for algorithm in self.results[env_name]:
                for result in self.results[env_name][algorithm]:
                    summary.append({
                        'Environment': env_name,
                        'Algorithm': algorithm,
                        'Seed': result['config']['seed'],
                        'Mean Reward': result['mean_reward'],
                        'Std Reward': result['std_reward']
                    })
        
        json_path = os.path.join(self.experiment_dir, 'experiment_results.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_numpy)


def main():
    parser = argparse.ArgumentParser(description='MACE-RL Experiment Runner')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    parser.add_argument('--run-experiments', action='store_true', help='Run all experiments')
    parser.add_argument('--run-ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--publication-mode', action='store_true', help='Publication-quality experiments')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel execution')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    print(f"\n🚀 MACE-RL Experiment Suite")
    print(f"📁 Results: {runner.experiment_dir}\n")
    
    if args.quick_test:
        print("🔬 Quick Test Mode\n")
        for env in EXPERIMENT_CONFIG['environments'][:2]:
            runner.run_single_experiment(env, 'mace_rl_full', 42, episode_limit=100)
    elif args.publication_mode:
        runner.run_publication_experiments(parallel=not args.no_parallel, ablation=args.run_ablation)
    elif args.run_experiments:
        runner.run_all_experiments(parallel=not args.no_parallel)
    elif args.run_ablation:
        runner.run_ablation_studies(parallel=not args.no_parallel)
    else:
        print("Use --help for options")
    
    print(f"\n✅ Complete! Results: {runner.experiment_dir}")


if __name__ == "__main__":
    main()
