# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:15:53 2025

@author: nollek

Main code to run an experiment
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import datetime as dt
import json
import sys
import torch
import wandb

#print("importing gym")
#import gym

import gymnasium as gym

import highway_env
highway_env.register_highway_envs()

from agent import config
from agent.sac import SACAgent
from agent.random import RandomAgent
from agent.ppo import PPO
from agent.drama import DramaAgent
from agent.evaluation import plot_learning_curve, average_fisher_sensitivity, plot_AFS_heatmap, performance_matrix, average_fisher_minmaxscale

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def main():
    
    # Command line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog="CL Parking Experiment",
                                     description="Run an experiment in the continual learning parking environment")
    
    parser.add_argument("-a", "--algorithm", 
                        help="Specify the RL algorithm to use.", 
                        choices=["random", "ppo", "sac", "drama", "ppo-ewc"],
                        required=True,
                        type=str)
    
    parser.add_argument("-s", "--scenario", 
                        help="Specify the continual learning setting to use."+
                              "perpendicular: single task perpendicular; "+
                              "diagonal-25: single task 25 deg diagonal; "+
                              "diagonal-50: single task 50 deg diagonal; "+
                              "parallel: single task parallel; "+
                              "parallel-adj: single task parallel with adjusted reward function; "+
                              "interleave: interleave scenarios uniformly; "+
                              "sequential-inc: encounter scenarios sequentially, increasing parking angle order; "+
                              "sequential-dec: encounter scenarios sequentially, decreasing parking angle order; ", 
                        choices=["perpendicular", "diagonal-25", "diagonal-50", "parallel",
                                 "parallel-adj", "interleave", "sequential-inc", "sequential-dec"],
                        required=True,
                        type=str)
    
    parser.add_argument("-p", "--path", 
                        help="Specify the directory path to store results.",
                        required=True,
                        type=str)
    
    parser.add_argument("-n", "--n-runs", 
                        help="Specify the number of times to repeat the experiment.",
                        default=5,
                        type=int)
    
    parser.add_argument("-g", "--gpu", 
                        help="Specify the GPU to use.",
                        default=0,
                        type=int)

    parser.add_argument("--start", 
                        help="Specify the run to start with. Use this if experiments crashed.",
                        default=0,
                        type=int)

    parser.add_argument("-d", "--desc", 
                        help="Description of the experiment.",
                        default="",
                        type=str)
    
    parser.add_argument("-k", "--wandb-key", 
                        help="Path to file containing API key for Weights and Biases.",
                        type=str)
    
    args, unknown = parser.parse_known_args()
    
    
    # Get all parameters needed for experiments
    # -------------------------------------------------------------------------
    
    # Get environment configuration
    if args.scenario == "perpendicular":
        env_params = config.env_perpendicular
    elif args.scenario == "diagonal-25":
        env_params = config.env_diagonal_25
    elif args.scenario == "diagonal-50":
        env_params = config.env_diagonal_50
    elif args.scenario == "parallel":
        env_params = config.env_parallel
    elif args.scenario == "parallel-adj":
        env_params = config.env_parallel_adj
    elif args.scenario == "interleave":
        env_params = config.env_interleave
    elif args.scenario == "sequential-inc":
        env_params = config.env_seq_perp_par
    elif args.scenario == "sequential-dec":
        env_params = config.env_seq_par_perp
     
    # Get RL agent hyperparameters
    if args.algorithm == "random":
        hyperparameters = {}
        AgentClass = RandomAgent
    elif args.algorithm == "ppo":
        hyperparameters = config.ppo_params
        AgentClass = PPO
    elif args.algorithm == "ppo-ewc":
        hyperparameters = config.ppo_ewc_params
        AgentClass = PPO
    elif args.algorithm == "sac":
        hyperparameters = config.sac_params
        #torch.autograd.set_detect_anomaly(True)
        AgentClass = SACAgent
    elif args.algorithm == "drama":
        hyperparameters = config.drama_params
        AgentClass = DramaAgent

    # Set WandB key if applicable
    if args.wandb_key is not None:
        with open(args.wandb_key) as f:
            key = f.read()
            wandb.login(key=key)
            
           
    # Set observation type based on RL algorithm
    if args.algorithm in ["random", "ppo", "sac"]:
        env_params.update({"observation": {"type": "CustomKinematicsGoal",
                                           "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                                           "scales": [100, 100, 5, 5, 1, 1],
                                           "normalize": False,
                                           }
                           })
    elif args.algorithm in ["drama"]:
        image_size = config.drama_params["config"]["BasicSettings"]["ImageSize"] # Ensure env image size matches image size in config
        print("image size" , image_size)
        env_params.update({"observation": {"type": "RGBObservation",
                                           "observation_shape": (175, 175), # Ensure whole parking lot is in the image
                                           "stack_size": 0, # No stacking of frames
                                           "center_ego": False,
                                           "image_size" : image_size # Observation image will be resized to (image_size, image_size)
                                           }
                           })
        
    # Get device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
        
    # Get training configuration
    train_params = config.train
    
    train_params["max_timesteps"] = train_params["scenario_timesteps"] * len(env_params["change_scenario"])
    
    if args.scenario == "parallel-adj":
        train_params["eval_scenarios"] += ["parallel-adj"]
        
    # Create output folder if missing
    if not os.path.exists(args.path):
        print(f"Creating output directory {args.path} ...")
        os.makedirs(args.path)
    
    # Write metadata of the experiment
    timestamp = dt.datetime.now()
    metadata = {"timestamp" : timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "cmd" : "python experiment.py " + ' '.join(sys.argv[1:]),
                "description" : args.desc,
                "agent_type" : args.algorithm,
                "scenario" : args.scenario,
                "output_path" : args.path,
                "n_runs" : args.n_runs,
                "random_baseline" : config.random_baseline_path,
                "training_params" : train_params,
                "hyperparameters" : hyperparameters,
                "environment_config" : env_params}
    
    with open(args.path +  f"/experiment-metadata_{round(timestamp.timestamp())}.json", "w") as outfile:
        json.dump(metadata, outfile, indent=2)
        
    # Remove 'scenario_timesteps' from training params because it is not a function argument
    del train_params["scenario_timesteps"]
    
    
    # Start experiments
    # -------------------------------------------------------------------------

    """
    # Create environment
    if False: #args.algorithm == "sac":
        # Create normalized vec env for stablebaseline implementation of SAC
        def create_env():
            env = gym.make('custom-parking-v0')
            env.configure(env_params)
            return env

        vec_env = DummyVecEnv([create_env])
        env = VecNormalize(vec_env)
    else:
        env = gym.make('custom-parking-v0')
        env.configure(env_params)
    """

    # Create environment
    env = gym.make('custom-parking-v0')
    env.configure(env_params)
    
    for i in range(args.start, args.n_runs):
        print(f"Starting run {i} ...")
        seed = i
        print("Using seed: ", seed)
        
        agent = AgentClass(env, device, seed=seed, **hyperparameters)
        
        agent.train(env, log_path=args.path, run_id=f"Run{i}", 
                    train_seed=seed, **train_params)
        
        agent.record_video(env, 5, args.path, 
                           run_id=i, seed=seed, deterministic=True)
        
        
    # Calculate aggregated metrics
    # -------------------------------------------------------------------------
    
    # Average Fisher Sensitivity
    average_fisher_sensitivity(args.path + "/data")
    average_fisher_minmaxscale(args.path + "/data")
    
    # Matrix for forward and backward transfer metrics
    if args.scenario in ["sequential-inc", "sequential-dec"]:
        for metric in ["reward", "success"]:
            performance_matrix(args.path + "/data", args.scenario, env_params["change_frequency"], performance_metric=metric)
    
    # Create plots
    # -------------------------------------------------------------------------
        
    # Get paths to data that should be plotted
    data_paths = {"random" : config.random_baseline_path}
    data_paths[args.algorithm] = args.path + "/data"
    
    # Get scenarios to plot. If experiment is single task experiment only plot
    # that parking scenario. Otherwise plot all
    if args.scenario in ["perpendicular", "diagonal-25", "diagonal-50", "parallel", "parallel-adj"]:
        scenarios = [args.scenario]
    else:
        scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]
    
    plot_learning_curve(plot_path=args.path + "/plots", 
                        data_paths=data_paths, 
                        max_steps=train_params["max_timesteps"], 
                        task_interval=env_params["change_frequency"],
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = scenarios
                        )
    
    # Create heatmaps for Average Fisher Sensitivity
    plot_AFS_heatmap(args.path, scale_all=True)
    
    
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)