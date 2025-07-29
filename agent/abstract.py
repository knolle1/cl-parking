# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:03:02 2025

Abstract agent class to define interface

@author: nollek
"""

from abc import abstractmethod

class AbstractAgent:

    @abstractmethod
    def __init__(self, env, device, seed, **hyperparameters):
        """
        Init hyperparameters and models
        """
        pass

    @abstractmethod
    def train(self, env, max_timesteps, eval_freq, n_eval_episodes, 
              eval_scenarios, log_path, run_id, train_seed, eval_seed,
              fisher_freq, fisher_steps):
        """
        Train agent in an environment for a certain number of steps. Evaluate
        the deterministic policy at regular intervals.
        
        env : gymnasium.Env
            Environment to train in
        max_timesteps : int
            Maximum number of timesteps to train in
        eval_freq : int
            Evaluate the agent every 'eval_freq' steps
        n_eval_episodes : int
            Number of episodes to evaluate the agent
        eval_scenarios : List of str
            List of scenarios to evaluate
        log_path : str
            Path to directory to store recorded metrics
        run_id : str
            Name of the run. This will be the header in the stored metrics
        train_seed : int
            Seed to initialise training environment
        eval_seed : int
            Seed to initialise evaluation environments
        fisher_steps : int
            Number of steps to roll out for calculating Fisher information
        """
        pass
    
    @abstractmethod
    def record_video(self, env, n_eval_episodes, log_path, run_id, 
                     seed=None, deterministic=False):
        pass