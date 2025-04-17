# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:03:16 2025

Random agent as a baseline

@author: nollek
"""

from .abstract import AbstractAgent
from .evaluation import Logger

import numpy as np
import copy
import gymnasium as gym
import os
from gymnasium.wrappers.record_video import RecordVideo

class RandomAgent(AbstractAgent):
    
    def __init__(self, env):
        """
        Init hyperparameters and models
        """
        # Get action space
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.shape = env.action_space.shape

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
        
        # Logging
        metrics = []
        for label in eval_scenarios:
            for m in ["reward_mean", "reward_std", 
                      "success_mean", "success_std",
                      "crashed_mean", "crashed_std",
                      "truncated_mean", "truncated_std",]:
                metrics.append(f"{label}_{m}")   

        logger = Logger(log_path, metrics)
        logger.open(run_id)
        
        # Set up evaluation environments
        eval_envs = {}
        for label in eval_scenarios:
            
            # Get evaluation environment configuration
            params = copy.copy(env.config)
            params.update(env.config["scenarios"][label])
            params["scenarios"] = {}       # Reset scenarios to just include current scenario
            params["change_scenario"] = []
            
            # Create evaluation environment
            env_name = env.unwrapped.spec.id
            eval_env = gym.make(env_name)
            eval_env.configure(params)
                
            eval_envs[label] = eval_env
        
        # Evaluate
        # +1 in range to evaluate step 0 as well
        for j in range((max_timesteps // eval_freq) + 1):
            step = eval_freq * j
            print(f"Evaluating step {step}")
            
            results = {}
            for label, env in eval_envs.items():
                # Seeding for evaluation purpose
                if eval_seed is not None:
                    env.np_random = np.random.default_rng(eval_seed)
                    env.action_space.seed(eval_seed)
                    env.observation_space.seed(eval_seed)
                    
                rewards_list = []
                success_list = []
                crashed_list = []
                truncated_list = []
                
                for i in range(n_eval_episodes):
                    
                    terminated = False
                    truncated = False
                    
                    obs, _ = env.reset()
                    
                    episode_reward = 0
                    
                    # Play out an episode
                    while not terminated and not truncated:
                        # Select random action
                        action = np.random.uniform(self.low, self.high, self.shape)
                        
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        
                    rewards_list.append(episode_reward)
                    success_list.append('is_success' in info.keys() and info['is_success'])
                    crashed_list.append('is_crashed' in info.keys() and info['is_crashed'])
                    truncated_list.append(truncated)
                
                results[f"{label}_reward_mean"] = np.mean(rewards_list)
                results[f"{label}_reward_std"] = np.std(rewards_list)
                results[f"{label}_success_mean"] = np.mean(success_list)
                results[f"{label}_success_std"] = np.std(success_list)
                results[f"{label}_crashed_mean"] = np.mean(crashed_list)
                results[f"{label}_crashed_std"] = np.std(crashed_list)
                results[f"{label}_truncated_mean"] = np.mean(truncated_list)
                results[f"{label}_truncated_std"] = np.std(truncated_list)
            
            logger.append(step, results)
            
        logger.close()
        
    def record_video(self, env, n_eval_episodes, log_path, run_id, 
                     seed=None, deterministic=False):
        
        # Create recording environment
        env_name = env.unwrapped.spec.id
        params = copy.copy(env.config)
        rec_env = gym.make(env_name, render_mode="rgb_array")
        rec_env.configure(params)
        
        # Create missing folders
        if not os.path.exists(log_path+"/video"):
            print(f"Creating output directory {log_path}/video")
            os.makedirs(log_path+"/video")
        
        rec_env = RecordVideo(rec_env, log_path+"/video", name_prefix=f"run-{run_id}",
                              video_callable=lambda episode_id: True, force=True)
        
        # Seeding for evaluation purpose
        if seed is not None:
            rec_env.np_random = np.random.default_rng(seed)
            rec_env.action_space.seed(seed)
            rec_env.observation_space.seed(seed)
        
        for i in range(n_eval_episodes):
            
            terminated = False
            truncated = False
            
            obs, _ = rec_env.reset()
            
            episode_reward = 0
            
            while not terminated and not truncated:
                action = np.random.uniform(self.low, self.high, self.shape)
                
                next_obs, reward, terminated, truncated, info = rec_env.step(action)
                episode_reward += reward
                
        rec_env.close()