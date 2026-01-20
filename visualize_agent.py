#!/usr/bin/env python3
"""
MACE-RL Agent Visualization
===========================

Visualize trained agents in GUI environments.

Usage:
    python visualize_agent.py --model-path path/to/model.pth --env CartPole-v1
    python visualize_agent.py --model-path path/to/model.pth --episodes 5
"""

import argparse
import os
import torch
import gym
import numpy as np
import time

from mace_rl.agents.ppo import PPO
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env


class AgentVisualizer:
    """Visualize trained agents in GUI environments."""
    
    def __init__(self, model_path, env_name="CartPole-v1"):
        self.model_path = model_path
        self.env_name = env_name
        self.env = None
        self.agent = None
        
        self._load_environment()
        self._load_model()
    
    def _load_environment(self):
        """Load the environment."""
        print(f"🎮 Loading environment: {self.env_name}")
        
        try:
            if 'MiniGrid' in self.env_name:
                self.env = make_minigrid_env(self.env_name)
            elif 'Bullet' in self.env_name:
                self.env = make_pybullet_env(self.env_name)
            else:
                self.env = gym.make(self.env_name, render_mode='human')
            print(f"✅ Environment loaded")
        except Exception as e:
            print(f"❌ Failed to load environment: {e}")
            raise
    
    def _load_model(self):
        """Load the trained model."""
        print(f"🤖 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Get dimensions
        if isinstance(self.env.observation_space, gym.spaces.Box):
            state_dim = self.env.observation_space.shape
            if len(state_dim) == 1:
                state_dim = state_dim[0]
        else:
            state_dim = self.env.observation_space.n
            
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            has_continuous_action_space = True
        else:
            action_dim = self.env.action_space.n
            has_continuous_action_space = False
        
        # Initialize and load agent
        self.agent = PPO(
            state_dim, action_dim,
            lr_actor=0.0003, lr_critic=0.001,
            gamma=0.99, K_epochs=4, eps_clip=0.2,
            has_continuous_action_space=has_continuous_action_space
        )
        
        try:
            self.agent.load(self.model_path)
            print(f"✅ Model loaded")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Using random policy")
    
    def play_episode(self):
        """Play one episode with visualization."""
        print(f"\n🎬 Playing episode...")
        
        episode_reward = 0
        
        reset_out = self.env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            if hasattr(self.env, 'render'):
                self.env.render()
            
            action = self.agent.select_action(state)
            
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            state = next_state
            episode_reward += reward
            step_count += 1
            time.sleep(0.05)
        
        print(f"✅ Completed - Steps: {step_count}, Reward: {episode_reward:.2f}")
        return episode_reward
    
    def run(self, num_episodes=3):
        """Run multiple episodes."""
        rewards = []
        for i in range(num_episodes):
            print(f"\n--- Episode {i + 1}/{num_episodes} ---")
            reward = self.play_episode()
            rewards.append(reward)
        
        print(f"\n📊 Summary:")
        print(f"   Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"   Best: {max(rewards):.2f}")
        
        return rewards


def main():
    parser = argparse.ArgumentParser(description='Visualize MACE-RL agents')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    
    args = parser.parse_args()
    
    visualizer = AgentVisualizer(args.model_path, args.env)
    visualizer.run(args.episodes)


if __name__ == "__main__":
    main()
