"""
🎛️ Beta Schedulers for Curiosity Reward Weighting
=================================================

This module provides various scheduling strategies for the beta parameter
that controls the balance between extrinsic and curiosity rewards:

Total Reward = Extrinsic Reward + β(t) × Curiosity Reward

Where β(t) changes over time according to different scheduling strategies.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BetaScheduler(ABC):
    """Abstract base class for beta scheduling strategies."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01, beta_max: float = 10.0):
        self.beta_initial = beta_initial
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.current_beta = beta_initial
        self.step_count = 0
        self.episode_count = 0
        
    @abstractmethod
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        """Update and return the current beta value."""
        pass
    
    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_beta = self.beta_initial
        self.step_count = 0
        self.episode_count = 0
    
    def get_beta(self) -> float:
        """Get current beta value."""
        return np.clip(self.current_beta, self.beta_min, self.beta_max)
    
    def get_info(self) -> Dict[str, Any]:
        """Get scheduler information for logging."""
        return {
            'scheduler_type': self.__class__.__name__,
            'current_beta': self.current_beta,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }


class ConstantScheduler(BetaScheduler):
    """Constant beta value throughout training."""
    
    def __init__(self, beta_value: float = 0.1):
        super().__init__(beta_initial=beta_value)
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
        return self.get_beta()


class LinearDecayScheduler(BetaScheduler):
    """Linear decay from initial to minimum value."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01, 
                 decay_episodes: int = 1000):
        super().__init__(beta_initial, beta_min)
        self.decay_episodes = decay_episodes
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        # Linear decay based on episodes
        decay_progress = min(self.episode_count / self.decay_episodes, 1.0)
        self.current_beta = self.beta_initial - (self.beta_initial - self.beta_min) * decay_progress
        return self.get_beta()


class ExponentialDecayScheduler(BetaScheduler):
    """Exponential decay scheduler."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01, 
                 decay_rate: float = 0.999):
        super().__init__(beta_initial, beta_min)
        self.decay_rate = decay_rate
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        self.current_beta = self.beta_initial * (self.decay_rate ** self.episode_count)
        return self.get_beta()


class CosineAnnealingScheduler(BetaScheduler):
    """Cosine annealing scheduler with optional restarts."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01,
                 period: int = 500, restart: bool = False):
        super().__init__(beta_initial, beta_min)
        self.period = period
        self.restart = restart
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        if self.restart:
            # With restarts
            cycle_progress = (self.episode_count % self.period) / self.period
        else:
            # Single cycle
            cycle_progress = min(self.episode_count / self.period, 1.0)
            
        # Cosine annealing formula
        cos_factor = (1 + math.cos(math.pi * cycle_progress)) / 2
        self.current_beta = self.beta_min + (self.beta_initial - self.beta_min) * cos_factor
        return self.get_beta()


class StepDecayScheduler(BetaScheduler):
    """Step-wise decay at specific episodes."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01,
                 decay_steps: list = None, decay_factor: float = 0.5):
        super().__init__(beta_initial, beta_min)
        self.decay_steps = decay_steps or [200, 500, 1000, 1500]
        self.decay_factor = decay_factor
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        # Count how many decay steps have been passed
        num_decays = sum(1 for step in self.decay_steps if self.episode_count >= step)
        self.current_beta = self.beta_initial * (self.decay_factor ** num_decays)
        return self.get_beta()


class AdaptiveScheduler(BetaScheduler):
    """Adaptive scheduler based on performance metrics."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01, beta_max: float = 2.0,
                 adaptation_rate: float = 0.1, performance_window: int = 100):
        super().__init__(beta_initial, beta_min, beta_max)
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.performance_history = []
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            
            # Keep only recent performance
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)
            
            # Adapt beta based on performance trend
            if len(self.performance_history) >= 20:  # Need some history
                recent_perf = np.mean(self.performance_history[-10:])
                older_perf = np.mean(self.performance_history[-20:-10])
                
                # If performance is improving, reduce curiosity weight
                # If performance is stagnating, increase curiosity weight
                if recent_perf > older_perf:
                    # Performance improving, reduce beta
                    self.current_beta *= (1 - self.adaptation_rate)
                else:
                    # Performance stagnating, increase beta
                    self.current_beta *= (1 + self.adaptation_rate)
                    
        return self.get_beta()


class WarmupCooldownScheduler(BetaScheduler):
    """Warmup phase followed by cooldown phase."""
    
    def __init__(self, beta_initial: float = 0.01, beta_peak: float = 1.0, beta_final: float = 0.01,
                 warmup_episodes: int = 200, cooldown_episodes: int = 800):
        super().__init__(beta_initial, beta_final, beta_peak)
        self.beta_peak = beta_peak
        self.beta_final = beta_final
        self.warmup_episodes = warmup_episodes
        self.cooldown_episodes = cooldown_episodes
        self.total_episodes = warmup_episodes + cooldown_episodes
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        if self.episode_count <= self.warmup_episodes:
            # Warmup phase: linear increase
            progress = self.episode_count / self.warmup_episodes
            self.current_beta = self.beta_initial + (self.beta_peak - self.beta_initial) * progress
        elif self.episode_count <= self.total_episodes:
            # Cooldown phase: linear decrease
            cooldown_progress = (self.episode_count - self.warmup_episodes) / self.cooldown_episodes
            self.current_beta = self.beta_peak - (self.beta_peak - self.beta_final) * cooldown_progress
        else:
            # After total episodes, keep final value
            self.current_beta = self.beta_final
            
        return self.get_beta()


class PerformanceBasedScheduler(BetaScheduler):
    """Beta scheduling based on performance thresholds."""
    
    def __init__(self, beta_initial: float = 1.0, beta_min: float = 0.01,
                 performance_thresholds: list = None, beta_values: list = None):
        super().__init__(beta_initial, beta_min)
        self.performance_thresholds = performance_thresholds or [50, 100, 200, 400]
        self.beta_values = beta_values or [1.0, 0.5, 0.2, 0.1]
        
        assert len(self.performance_thresholds) == len(self.beta_values), \
            "Thresholds and beta values must have same length"
    
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        if performance_metric is not None:
            # Find appropriate beta based on performance
            for i, threshold in enumerate(self.performance_thresholds):
                if performance_metric >= threshold:
                    self.current_beta = self.beta_values[i]
                    break
            else:
                # If no threshold met, use initial beta
                self.current_beta = self.beta_initial
                
        return self.get_beta()


class CyclicalScheduler(BetaScheduler):
    """Cyclical beta scheduler that oscillates between values."""
    
    def __init__(self, beta_min: float = 0.01, beta_max: float = 1.0,
                 cycle_length: int = 200, cycle_type: str = 'triangular'):
        super().__init__(beta_max, beta_min, beta_max)
        self.cycle_length = cycle_length
        self.cycle_type = cycle_type
        
    def update(self, episode: int = None, step: int = None, performance_metric: float = None) -> float:
        if episode is not None:
            self.episode_count = episode
        if step is not None:
            self.step_count = step
            
        cycle_position = (self.episode_count % self.cycle_length) / self.cycle_length
        
        if self.cycle_type == 'triangular':
            # Triangular wave
            if cycle_position <= 0.5:
                # Ascending
                factor = cycle_position * 2
            else:
                # Descending
                factor = 2 * (1 - cycle_position)
                
        elif self.cycle_type == 'sinusoidal':
            # Sinusoidal wave
            factor = (math.sin(2 * math.pi * cycle_position) + 1) / 2
            
        else:  # sawtooth
            factor = cycle_position
            
        self.current_beta = self.beta_min + (self.beta_max - self.beta_min) * factor
        return self.get_beta()


# Factory function for easy scheduler creation
def create_beta_scheduler(scheduler_type: str, **kwargs) -> BetaScheduler:
    """Factory function to create beta schedulers."""
    schedulers = {
        'constant': ConstantScheduler,
        'linear_decay': LinearDecayScheduler,
        'exponential_decay': ExponentialDecayScheduler,
        'cosine_annealing': CosineAnnealingScheduler,
        'step_decay': StepDecayScheduler,
        'adaptive': AdaptiveScheduler,
        'warmup_cooldown': WarmupCooldownScheduler,
        'performance_based': PerformanceBasedScheduler,
        'cyclical': CyclicalScheduler
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Available types: {list(schedulers.keys())}")
    
    return schedulers[scheduler_type](**kwargs)


# Predefined scheduler configurations for common use cases
SCHEDULER_PRESETS = {
    'conservative': {
        'scheduler_type': 'constant',
        'beta_value': 0.1
    },
    'aggressive_decay': {
        'scheduler_type': 'exponential_decay',
        'beta_initial': 2.0,
        'beta_min': 0.01,
        'decay_rate': 0.995
    },
    'gentle_decay': {
        'scheduler_type': 'linear_decay',
        'beta_initial': 1.0,
        'beta_min': 0.1,
        'decay_episodes': 1500
    },
    'exploration_first': {
        'scheduler_type': 'warmup_cooldown',
        'beta_initial': 0.01,
        'beta_peak': 2.0,
        'beta_final': 0.1,
        'warmup_episodes': 300,
        'cooldown_episodes': 1200
    },
    'adaptive_smart': {
        'scheduler_type': 'adaptive',
        'beta_initial': 0.5,
        'beta_min': 0.01,
        'beta_max': 2.0,
        'adaptation_rate': 0.05
    },
    'cyclical_exploration': {
        'scheduler_type': 'cyclical',
        'beta_min': 0.05,
        'beta_max': 0.8,
        'cycle_length': 400,
        'cycle_type': 'triangular'
    }
}


def get_preset_scheduler(preset_name: str) -> BetaScheduler:
    """Get a predefined scheduler configuration."""
    if preset_name not in SCHEDULER_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available presets: {list(SCHEDULER_PRESETS.keys())}")
    
    config = SCHEDULER_PRESETS[preset_name].copy()
    scheduler_type = config.pop('scheduler_type')
    return create_beta_scheduler(scheduler_type, **config)

