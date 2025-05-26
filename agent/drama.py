# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:03:02 2025

Drama implementation.

@author: nollek
"""

from .abstract import AbstractAgent
from .evaluation import Logger

from .drama_lib.train import (
    #train_world_model_step,
    #world_model_imagine_data,
    joint_train_world_model_agent,
    build_world_model,
    build_agent,
    DotDict,
    parse_args_and_update_config,
    update_model_parameters,
    )
from .drama_lib.utils import seed_np_torch, WandbLogger
from .drama_lib.replay_buffer import ReplayBuffer

from gymnasium.spaces import Box, Discrete

import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os
import pandas as pd
from line_profiler import profile
import warnings
import ast


class DramaAgent(AbstractAgent):

    def __init__(self, env, device, config):
        """
        Init hyperparameters and models
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings("ignore")
        
        # Map string dtype to torch dtype
        def map_dtype(d):
            for key, value in d.items():
                if isinstance(d[key], dict):
                    map_dtype(d[key])
                else:
                    if key == 'dtype':
                        dtype_map = {
                            'float32': torch.float32,
                            'float16': torch.float16,
                            'bfloat16': torch.bfloat16
                        }
                        d[key] = dtype_map[value]
        map_dtype(config)
    
        self.config = DotDict(config)
        self.device = device
        
        if isinstance(env.action_space, Box):
            self.continous_action = True
            action_dim = np.prod(env.action_space.shape)
            action_shape = env.action_space.shape # For replay buffer
        elif isinstance(env.action_space, Discrete):
            self.continous_action = False
            action_dim = env.action_space.n
            action_shape = ()  # For replay buffer. Scalar shape for discrete actions
        else:
            raise AssertionError(f"action space is not valid {env.action_space}")

        seed_np_torch(seed=self.config.BasicSettings.Seed)

        # build world model and agent
        self.world_model = build_world_model(self.config, action_dim, device=self.device,is_continuous=self.continous_action)
        self.agent = build_agent(self.config, action_dim, device=self.device,is_continuous=self.continous_action)
        update_model_parameters(self.config, self.world_model, self.agent)
        
        if (self.config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
            self.world_model = torch.compile(self.world_model)
            self.agent = torch.compile(self.agent)
            
        if self.config.BasicSettings.SavePath != 'None':
            print('Loading models')
            self.world_model.load_state_dict(torch.load(f"{self.config.BasicSettings.SavePath}/world_model.pth"))
            self.agent.load_state_dict(torch.load(f"{self.config.BasicSettings.SavePath}/agent.pth"))
    
        self.logger = WandbLogger(config=self.config, project=self.config.Wandb.Init.Project, mode=self.config.Wandb.Init.Mode)
        self.logdir = f"./saved_models/{self.config.n}/{self.config.BasicSettings.Env_name}/{self.logger.run.id}"

        # build replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config,
            action_shape,
            device=self.device
        )

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
        
        # train
        joint_train_world_model_agent(env, self.continous_action, self.config, self.logdir, self.replay_buffer, 
                                      self.world_model, self.agent, self.logger)
        self.logger.close()
        
        pass
    
        
    def record_video(self, env, n_eval_episodes, log_path, run_id, 
                     seed=None, deterministic=False):
        pass
