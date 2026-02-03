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
from .drama_lib.env_wrapper import MaxLast2FrameSkipWrapper

from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers.record_video import RecordVideo

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
import copy


class DramaAgent(AbstractAgent):

    def __init__(self, env, device, seed, config):
        """
        Init hyperparameters and models
        """
    
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings("ignore")

        #print(env.config["observation"]["features"].index("sin_h"))
        
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

        # Update to ensure inputs for encoder/decoder match observation image size
        img_size = self.config.BasicSettings.ImageSize
        self.config.update_or_create("Models.WorldModel.Encoder.InputSize", [3, img_size, img_size])
        self.config.update_or_create("Models.WorldModel.Decoder.InputSize", [3, img_size, img_size])
        
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

        seed_np_torch(seed=seed)

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
        
        self.config.update_or_create("BasicSettings.Seed", train_seed)
        self.config.update_or_create("JointTrainAgent.SampleMaxSteps", max_timesteps + 5000)
        self.config.update_or_create("JointTrainAgent.FreezeWorldModelAfterSteps", max_timesteps)
        self.config.update_or_create("JointTrainAgent.FreezeBehaviourAfterSteps", max_timesteps)
        self.config.update_or_create("Evaluate.EverySteps", eval_freq)
        self.config.update_or_create("Evaluate.EpisodeNum", n_eval_episodes)
        
        """
        self.config.BasicSettings.Seed = train_seed
        self.config.JointTrainAgent.SampleMaxSteps = max_timesteps + 5000
        self.config.JointTrainAgent.FreezeWorldModelAfterSteps = max_timesteps
        self.config.JointTrainAgent.FreezeBehaviourAfterSteps = max_timesteps
        self.config.Evaluate.EverySteps = eval_freq
        self.config.Evaluate.EpisodeNum = n_eval_episodes
        """
        
        # Weights & Biases logging
        logger_wandb = WandbLogger(config=self.config, project=self.config.Wandb.Init.Project, mode=self.config.Wandb.Init.Mode,dir=log_path)
        logdir_wandb = f"{log_path}/saved_models/{self.config.n}/{self.config.BasicSettings.Env_name}/{logger_wandb.run.id}"
        
        # CSV logging
        metrics = []
        for label in eval_scenarios:
            for m in ["reward_mean", "reward_std", 
                      "success_mean", "success_std",
                      "crashed_mean", "crashed_std",
                      "truncated_mean", "truncated_std",]:
                metrics.append(f"{label}_{m}")   

        logger_csv = Logger(log_path, metrics)
        logger_csv.open(run_id)
        
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
            #eval_env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
            eval_env = gymnasium.make(env_name, render_mode="rgb_array")
            eval_env.configure(params)
            #eval_env = MaxLast2FrameSkipWrapper(eval_env, skip=4)
            #eval_env = gymnasium.wrappers.ResizeObservation(eval_env, shape=self.config.BasicSettings.ImageSize)
            
            # Seeding for evaluation purpose
            if eval_seed is not None:
                eval_env.np_random = np.random.default_rng(eval_seed)
                eval_env.action_space.seed(eval_seed)
                eval_env.observation_space.seed(eval_seed)
                
            eval_envs[label] = eval_env
        
        # train
        joint_train_world_model_agent(env, self.continous_action, self.config, logdir_wandb, self.replay_buffer, 
                                      self.world_model, self.agent, logger_wandb, logger_csv, eval_envs, eval_seed)
        
        logger_wandb.close()
        logger_csv.close()
        
    
        
    def record_video(self, env, n_eval_episodes, log_path, run_id, 
                     seed=None, deterministic=False):
        # Create recording environment
        config = env.config
        env_name = env.unwrapped.spec.id
        #env_name = env.get_attr("unwrapped")[0].spec.id
        #config = env.get_attr("config")[0]
        params = copy.copy(config)
        rec_env = gymnasium.make(env_name, render_mode="rgb_array")
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
            
            current_obs, _ = env.reset()
            context_obs = deque(maxlen=self.config.JointTrainAgent.RealityContextLength)
            context_action = deque(maxlen=self.config.JointTrainAgent.RealityContextLength)
            
            episode_reward = 0
            
            while not terminated and not truncated:
                with torch.no_grad():
                    if len(context_action) == 0:
                        action = rec_env.action_space.sample()
                        # action = np.array([action], dtype=int)
                        # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                    else:
                        context_latent = self.world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                        model_context_action = np.stack(list(context_action))
                        #model_context_action = torch.Tensor(model_context_action).to(world_model.device)
                        # current_obs_tensor = rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255
                        
                        if self.is_continuous:
                            model_context_action = rearrange(torch.Tensor(model_context_action).to(world_model.device), "L A-> 1 L A")
                        else:
                            model_context_action = rearrange(torch.Tensor(model_context_action).to(world_model.device), "L -> 1 L")
                            
                        #print(model_context_action.shape)
                        if self.world_model.model == 'Transformer':
                            prior_flattened_sample, last_dist_feat = self.world_model.calc_last_dist_feat(context_latent, model_context_action)
                            # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                        elif self.world_model.model == 'Mamba' or self.world_model.model == 'Mamba2':
                            # prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent[:,-1:], model_context_action[:,-1:], inference_params)
                            prior_flattened_sample, last_dist_feat = self.world_model.calc_last_dist_feat(context_latent, model_context_action)
                            # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                        action = self.agent.sample_as_env_action(
                            torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                            greedy=True
                        )[0]

                context_obs.append(rearrange(torch.Tensor(current_obs).to(self.world_model.device), "H W C -> 1 1 C H W")/255)
                context_action.append(action)

                obs, reward, terminated, truncated, info = rec_env.step(action)
                # cv2.imshow("current_obs", process_visualize(obs[0]))
                # cv2.waitKey(10)
                # update current_obs, current_info and sum_reward
                episode_reward += reward
                current_obs = obs
