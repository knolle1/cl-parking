# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:18:33 2025

Wrapper for stablebaseline3 SAC implementation

@author: nollek
"""

from .abstract import AbstractAgent
from .evaluation import Logger

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.wrappers.record_video import RecordVideo  # Older version
#from gymnasium.wrappers.rendering import RecordVideo

import copy
import gymnasium as gym
import json
import numpy as np
import os
import torch


class EvalCallback(BaseCallback):
    """
    Callback for evaluation during training

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    
    def __init__(self, env, eval_freq, n_eval_episodes, logger, scenarios, run_id, 
                 seed=None, verbose: int = 0, fisher_freq=None, fisher_steps=1024):
        super().__init__(verbose)
        
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.my_logger = logger
        self.run_id = run_id
        self.env = env
        self.fisher_freq = fisher_freq
        self.fisher_steps = fisher_steps
        
        # Set up evaluation environments
        self.eval_envs = {}
        for label in scenarios:
            
            # Get evaluation environment configuration
            params = copy.copy(env.config)
            params.update(env.config["scenarios"][label])
            params["scenarios"] = {}       # Reset scenarios to just include current scenario
            params["change_scenario"] = []
            
            # Create evaluation environment
            env_name = env.unwrapped.spec.id
            eval_env = gym.make(env_name)
            eval_env.configure(params)
            
            # Seeding for evaluation purpose
            if self.seed is not None:
                eval_env.np_random = np.random.default_rng(seed)
                eval_env.action_space.seed(seed)
                eval_env.observation_space.seed(seed)
                
            self.eval_envs[label] = eval_env
            
            
    def evaluate(self):
        """
        Evaluate performance in each scenario 'n_eval_episodes' times
        """
        results = {}
        for label, env in self.eval_envs.items():
            # Seeding for evaluation purpose
            if self.seed is not None:
                env.np_random = np.random.default_rng(self.seed)
                env.action_space.seed(self.seed)
                env.observation_space.seed(self.seed)
        
            rewards_list = []
            success_list = []
            crashed_list = []
            truncated_list = [] 
            
            for i in range(self.n_eval_episodes):
                
                terminated = False
                truncated = False
                
                obs, _ = env.reset()
                
                episode_reward = 0
                while not terminated and not truncated:
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
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
        
        self.my_logger.append(self.n_calls, results)
        return results
        
        
    def calculate_fisher(self):
        """
        Calculate Fisher information matrix when task changes
        """
        # Create copy of environment for current task
        task = self.env.get_current_task()
        print("Current Task\n----------------------------------")
        print(task)
        
        # Get evaluation environment configuration
        params = copy.copy(self.env.config)
        params["change_scenario"] = [[task["labels"], task["probs"]]] 
        
        # Create evaluation environment
        env_name = self.env.unwrapped.spec.id
        eval_env = gym.make(env_name)
        eval_env.configure(params)
        
        # Seeding for evaluation purpose
        if self.seed is not None:
            eval_env.np_random = np.random.default_rng(self.seed)
            eval_env.action_space.seed(self.seed)
            eval_env.observation_space.seed(self.seed)
        
        
        # Init fisher information matrix. Note: this is the diagonal of the
        # matrix for each parameter
        fisher = {n: torch.zeros_like(p).to(p.device) for n, p in 
                  self.model.actor.named_parameters()} 

        obs,_ = eval_env.reset()

        # Calculate diagonals of fisher information matrix
        # Rollout experience and for each datapoint (i.e. step) calculate gradient
        # and approximate fisher
        for _ in range(self.fisher_steps):            
            
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Normalise observations and reward because StableBaseline3 ReplayBuffer normalises these when sampling the buffer
            # Stablebaseline3 SAC uses VecNormalize wrapper around training env - maintains running mean and std for normalization
            # Use same normalization as training environment to calculate same loss gradients as during training
            vec_norm_env = self.model.get_vec_normalize_env()
            if vec_norm_env is not None:
                obs = vec_norm_env.normalize_obs(obs)
                next_obs = vec_norm_env.normalize_obs(next_obs)
                reward = vec_norm_env.normalize_reward(reward).astype(np.float32)
            
            #obs = eval_env.normalize_obs(obs)
            #next_obs = eval_env.normalize_obs(next_obs)
            #reward = eval_env.normalize_reward(reward).astype(np.float32)
            
            # Convert to tensors
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            
            done = int(terminated or truncated)
            
            # Reset gradients
            self.model.actor.optimizer.zero_grad()
            
            # Code from stablebaseline3 for calculating loss
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py
            # -------------------------------------------------------------------------
            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.model.actor.action_log_prob(obs_tensor)
            log_prob = log_prob.reshape(-1, 1)
            
            if self.model.ent_coef_optimizer is not None and self.model.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.model.log_ent_coef.detach())
            else:
                ent_coef = self.model.ent_coef_tensor
            
            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.model.actor.action_log_prob(next_obs_tensor)
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.model.critic_target(next_obs_tensor, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = reward + (1 - done) * self.model.gamma * next_q_values
                
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.model.critic(obs_tensor, action_tensor)
            
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = torch.cat(self.model.critic(obs_tensor, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            # -------------------------------------------------------------------------
                        
            # Calculate gradients
            actor_loss.backward()
                        
            # Calculate matrix diagonals using approximation of second derivative
            # Diagonal of fisher information matrix for parameter i:
            # F_i = E[(\partial J(theta)/ \partial theta_i)^2]
            for n, p in self.model.actor.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2) / self.fisher_steps
                    
            if done:
                obs,_ = eval_env.reset()
                
        eval_env.close()
        
        # Get path to save diagonals
        path = self.my_logger.data_path + f"/fisher-information_step-{self.n_calls}"
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save as JSON
        with open(path +  f"/fisher-diagonal_{self.run_id}.json", "w") as outfile:
            json.dump({k: v.tolist() for k, v in fisher.items()}, outfile)


    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        self.my_logger.open(self.run_id)
        self.evaluate()

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        
        if self.n_calls % self.eval_freq == 0:
            print(f"Evaluating step {self.n_calls}")
            self.evaluate()
            
        if self.fisher_freq is not None:
            if self.n_calls % self.fisher_freq == 0:
                print(f"Calculating Fisher step {self.n_calls}")
                self.calculate_fisher()
            
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        # Evaluate final agent
        print(f"Evaluating step {self.n_calls}")
        self.evaluate()
            
        if self.fisher_freq is not None:
            print(f"Calculating Fisher step {self.n_calls}")
            self.calculate_fisher()
        
        print("Closing logger")
        for label, env in self.eval_envs.items():
            env.close()
        self.my_logger.close()


class SACAgent(AbstractAgent):
    
    def __init__(self, env, device, buffer_size, learning_rate, gamma, batch_size, tau,
                 num_layers, layer_size, verbose=1):
        """
        Init hyperparameters and models
        """
        
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_kwargs = dict(net_arch=[layer_size] * num_layers)
        
        self.model = SAC('MlpPolicy', 
                         env,
                         device=device,
                         verbose=verbose, 
                         buffer_size=self.buffer_size,
                         learning_rate=self.learning_rate,
                         gamma=self.gamma, 
                         batch_size=self.batch_size, 
                         tau=self.tau,
                         policy_kwargs=self.policy_kwargs)
        


    def train(self, env, max_timesteps, eval_freq, n_eval_episodes, 
              eval_scenarios, log_path, run_id, train_seed, eval_seed,
              verbose=1, fisher_freq=None, fisher_steps=1024):
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
        """
        
        # Reset environment global step, which is used to keep track for 
        # task changes
        env.reset_global_step()
        
        if train_seed is not None:
            env.np_random = np.random.default_rng(train_seed)
            env.action_space.seed(train_seed)
            env.observation_space.seed(train_seed)
            
        self.model.set_env(env)
        
        # Logging
        metrics = []
        for label in eval_scenarios:
            for m in ["reward_mean", "reward_std", 
                      "success_mean", "success_std",
                      "crashed_mean", "crashed_std",
                      "truncated_mean", "truncated_std",]:
                metrics.append(f"{label}_{m}")   

        logger = Logger(log_path, metrics)

        callback = EvalCallback(env=env, 
                                eval_freq=eval_freq, 
                                n_eval_episodes=n_eval_episodes, 
                                logger=logger, 
                                scenarios=eval_scenarios, 
                                run_id=run_id, 
                                seed=eval_seed, 
                                verbose=verbose,
                                fisher_freq=fisher_freq,
                                fisher_steps=fisher_steps)
        
        self.model.learn(max_timesteps,callback=callback)
        
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
                              episode_trigger=lambda x: True)
        
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
                action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = rec_env.step(action)
                episode_reward += reward
                
        rec_env.close()
