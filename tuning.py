# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:34:04 2025

Script for hyperparameter tuning

@author: nollek
"""

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import datetime as dt
import json
import sys
import wandb
import pandas as pd
import torch

import gymnasium as gym
import highway_env
highway_env.register_highway_envs()

from agent import config
from agent.sac import SACAgent
from agent.ppo import PPO

# Global run counter
run_counter = 0

def main():
    global run_counter
    
    # Command line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(prog="CL Parking Hyperparameter Tuning",
                                     description="Run hyperparameter tuning in the continual learning parking environment")
    
    parser.add_argument("-a", "--algorithm", 
                        help="Specify the RL algorithm to use.", 
                        choices=["ppo", "sac", "drama"],
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
                        default="perpendicular",
                        type=str)
    
    parser.add_argument("-p", "--path", 
                        help="Specify the directory path to store results.",
                        required=True,
                        type=str)
    
    parser.add_argument('--login', 
                        help='Path to file with Weights & Bias API login', 
                        default=None,
                        type=str)
    
    parser.add_argument('--project', 
                        help='Weights & Bias project name', 
                        default="test",
                        type=str)
    
    parser.add_argument("-n", "--n-runs", 
                        help="Specify the maximum number of hyperparameters to test.",
                        default=50,
                        type=int)
    
    parser.add_argument('--sweep', 
                        help='Weights & Bias sweep ID. Use this to continue existing sweeps', 
                        default=None,
                        type=str)
    
    parser.add_argument('--run-counter', 
                        help='Init value for run counter. Set this when passing sweep ID to avoid run number conflicts', 
                        default=0,
                        type=int)
    
    args, unknown = parser.parse_known_args()
    
    if args.login is None:
        wandb.login()
    else:
        if os.path.isfile(args.login):
            with open(args.login, "r") as f:
                api_key = f.read()
        else:
            api_key = args.login
            
        wandb.login(key=api_key)
        
    run_counter = args.run_counter
    
    
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
        
    if args.scenario in ["perpendicular", "diagonal-25", "diagonal-50", 
                         "parallel", "parallel-adj"]:
        eval_scenarios = [args.scenario]
    elif args.scenario == "interleave":
        eval_scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]
        
    # Get RL agent hyperparameters
    if args.algorithm == "ppo":
        hp_bounds = {'actor_lr': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-3},
                     'critic_lr': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-3},
                     'memory_size': {'values': [1024, 2048, 4096]},
                     'K_epochs': {'values': [5, 10, 20]},
                     'gamma': {'distribution': 'uniform', 'min': 0.95, 'max': 0.999},
                     'lamb': {'distribution': 'uniform', 'min': 0.90, 'max': 0.99},
                     'early_stop': {'value': False},
                     'cal_total_loss': {'value': True},
                     'parameters_hardshare': {'value': False},
                     'c1': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
                     'c2': {'distribution': 'uniform', 'min': 0, 'max': 0.1},
                     'minibatch_size': {'values': [32, 64, 128, 256, 512, 1024]},
                     'kl_threshold': {'value': 0.15},
                     'max_grad_norm': {'values': [0.5, 1.0]},
                     'eps_clip': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
                     'num_cells': {'values': [64, 128, 256, 512, 1024]},
                     'layer_num': {'values': [2, 3, 4, 5, 6]},
                     }
        AgentClass = PPO
        
    elif args.algorithm == "sac":
        hp_bounds = {"buffer_size" : {'distribution': 'int_uniform', 'min': 100_000, 'max': 1_500_000}, 
                     "learning_rate" : {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-3}, 
                     "gamma" : {'distribution': 'uniform', 'min': 0.95, 'max': 0.999},
                     "batch_size" : {'values': [32, 64, 128, 256, 512, 1024]}, 
                     "tau" : {'distribution': 'uniform', 'min': 0, 'max': 1},
                     "num_layers" : {'values': [2, 3, 4, 5, 6]}, 
                     "layer_size" : {'values': [64, 128, 256, 512, 1024]}, 
                     "verbose" : {'value': 0},
                      }
        AgentClass = SACAgent
        
    elif args.algorithm == "drama":
        hp_bounds = {}
        # TODO: AgentClass = DramaAgent
        
    # Get training configuration
    train_params = {"max_timesteps" : 250_000, # Number of timesteps to train in total
                    "eval_freq" : 10_000,      # Evaluate the policy every 'eval_freq' steps
                    "n_eval_episodes" : 30,    # Number of episodes to run per evaluation
                    "eval_scenarios" : eval_scenarios, 
                    "eval_seed" : 1234,
                    "fisher_freq" : None, # No evaluation of Fisher information matrix
                    "fisher_steps" : 0}   
    
    # Define search space
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'reward'},
        'parameters': {'hyperparams' : {'parameters' : hp_bounds},
                       'env_params': {'value': env_params},
                       'train_params' : {'value': train_params}
                       },
        'early_terminate': {'type': 'hyperband',
                            'min_iter': 20,
                            'max_iter': 100,
                            's': 2
                            }
        }
    
    # Set sweep name
    sweep_configuration['name'] = f"{args.algorithm}-{args.scenario}-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"

    
    print(sweep_configuration)
    
    # Create output folder if missing
    if not os.path.exists(args.path):
        print(f"Creating output directory {args.path} ...")
        os.makedirs(args.path)
    
    # Write metadata of the experiment
    timestamp = dt.datetime.now()
    metadata = {"timestamp" : timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "cmd" : "python tuning.py " + ' '.join(sys.argv[1:]),
                "agent_type" : args.algorithm,
                "scenario" : args.scenario,
                "output_path" : args.path,
                "n_runs" : args.n_runs,
                "random_baseline" : config.random_baseline_path,
                "training_params" : train_params,
                "hp_bounds" : hp_bounds,
                "environment_config" : env_params}
    
    with open(args.path +  f"/experiment-metadata_{round(timestamp.timestamp())}.json", "w") as outfile:
        json.dump(metadata, outfile, indent=2)
        
    # Create file for logging hyperparams if file does not yet exist
    if not os.path.isfile(args.path +  f"/hyperparameters.csv"):
        hyperparam_names = [x for x in hp_bounds.keys()]
        hyperparam_names.sort()
        with open(args.path +  f"/hyperparameters.csv", "w") as outfile:
            outfile.write(",".join(["RunID"] + hyperparam_names) + "\n")

    
    
    
    # Function to optimise
    # -------------------------------------------------------------------------
    
    # Function to execute a training run 
    def sweep_function():
        global run_counter
        
        # Init run
        run = wandb.init(group=f"hyperparameters-{sweep_configuration['name']}")
        run.name = f"{sweep_configuration['name']}_run-{run_counter}"
        
        #run.group = f"hyperparameter-tuning-{sweep_configuration['name']}"
        
        env_params = wandb.config.env_params
        hyperparameters = wandb.config.hyperparams
        train_params = wandb.config.train_params
        
        # Log hyperparameters
        hyperparam_names = [x for x in hyperparameters.keys()]
        hyperparam_names.sort()
        hyperparam_values = [str(hyperparameters[x]) for x in hyperparam_names]
        with open(args.path +  f"/hyperparameters.csv", "a") as outfile:
            outfile.write(",".join([f"Run{run_counter}"] + hyperparam_values) + "\n")
        
        # Create environment
        env = gym.make('custom-parking-v0')
        env.configure(env_params)

        # Get device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Train hyperparameters
        print(f"Starting run {run_counter} ...")
        agent = AgentClass(env, device, seed=run_counter, **hyperparameters)
        agent.train(env, log_path=args.path, run_id=f"Run{run_counter}", 
                        train_seed=run_counter, **train_params)
            
        agent.record_video(env, 1, args.path, run_id=run_counter, seed=run_counter, deterministic=True)
        
        # Save metrics to wandb
        df_merge = None
        for metric in ["reward", "success", "crashed", "truncated"]:
            save_path = args.path + "/data" + f"/{args.scenario}_{metric}_mean.csv"
            df = pd.read_csv(save_path, usecols=["idx", f"Run{run_counter}"])
            
            df = df.rename(columns={f"Run{run_counter}" : metric})
            
            # Convert all columns to int/float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            if df_merge is None:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df.drop(columns=["idx"]), left_index=True, right_index=True)
        
        for _, data in df_merge.to_dict('index').items():
            step = data["idx"]
            del data["idx"]
            wandb.log(data, step=step)
            
        # Save video to wandb
        video_path = args.path + f"/video/run-{run_counter}-episode-0.mp4"
        wandb.log({f"Episode_recording_{args.scenario}": wandb.Video(video_path, format="mp4")}, step=step)#train_params["max_timesteps"])
        
        run_counter += 1
        
    # Run the sweep
    if args.sweep is not None:
        sweep_id = args.sweep
    else:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
    wandb.agent(sweep_id, function=sweep_function, count=args.n_runs)
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)