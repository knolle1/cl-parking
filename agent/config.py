# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:29:08 2025

@author: nollek
"""
import copy 

# General training configuration
train = {"scenario_timesteps" : 500_000, # Number of timesteps to train per scenario
         "eval_freq" : 10_000,           # Evaluate the policy every 'eval_freq' steps
         "n_eval_episodes" : 30,         # Number of episodes to run per evaluation
         "eval_scenarios" : ["perpendicular", "diagonal-25", "diagonal-50", "parallel"], 
         "eval_seed" : 1234,
         "fisher_freq" : 500_000, # Evaluate the Fisher information matrix every 'fisher_freq' steps
         "fisher_steps" : 1024}   # Number of experiences to generate for Fisher information calculation

random_baseline_path = "results/random_baseline/data"

# Highway-env configuration
env_default = {"add_width" : [0,0],
               "fixed_goal" : [[0, 3], [0, -4], [1, 3], [1, -4]],
               "start_pos" : [[-35, 0], [35, 3.141592653589793]],
               "success_goal_reward": 0.12,
               "collision_reward": -10,
               "reward_p" : 0.5,
               "collision_reward_factor" : 50,
               "scenarios" : {"perpendicular" : {"parking_angles" : [0, 0],
                                                 "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05],
                                                 "adjust_heading" : False
                                                 },
                              "diagonal-25" : {"parking_angles" : [25, 25],
                                               "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05],
                                               "adjust_heading" : False
                                               },
                              "diagonal-50" : {"parking_angles" : [50, 50],
                                               "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05],
                                               "adjust_heading" : False
                                               },
                              "parallel" : {"parking_angles" : [90, 90],
                                            "reward_weights": [0.3, 1, 0, 0, 0.05, 0.05],
                                            "adjust_heading" : False
                                            },
                              "parallel-adj" : {"parking_angles" : [90, 90],
                                                "reward_weights": [0.3, 1, 0, 0, 0.05, 0.05],
                                                "adjust_heading" : True
                                                }
                              },
               "change_scenario" : [[["perpendicular"], [1]]],
               "change_frequency" : train["scenario_timesteps"] # Changing scenarios corresponds to training configuration
               }

env_perpendicular = copy.copy(env_default)
env_perpendicular["change_scenario"] = [[["perpendicular"], [1]]]

env_diagonal_25 = copy.copy(env_default)
env_diagonal_25["change_scenario"] = [[["diagonal-25"], [1]]]

env_diagonal_50 = copy.copy(env_default)
env_diagonal_50["change_scenario"] = [[["diagonal-50"], [1]]]

env_parallel =  copy.copy(env_default)
env_parallel["change_scenario"] = [[["parallel"], [1]]]

env_parallel_adj = copy.copy(env_default)
env_parallel_adj["change_scenario"] = [[["parallel-adj"], [1]]]

env_interleave = copy.copy(env_default)
env_interleave["change_scenario"] = [[["perpendicular", "diagonal-25", "diagonal-50", "parallel"], [0.25, 0.25, 0.25, 0.25]]]

env_seq_perp_par = copy.copy(env_default)
env_seq_perp_par["change_scenario"] = [[["perpendicular"], [1]],
                                       [["diagonal-25"], [1]],
                                       [["diagonal-50"], [1]],
                                       [["parallel"], [1]]]

env_seq_par_perp = copy.copy(env_default)
env_seq_par_perp["change_scenario"] = [[["parallel"], [1]],
                                       [["diagonal-50"], [1]],
                                       [["diagonal-25"], [1]],
                                       [["perpendicular"], [1]]]

# PPO Hyperparameters
ppo_params = {}

# SAC Hyperparameters
sac_params = {"buffer_size" : int(1e6), 
              "learning_rate" : 1e-3, 
              "gamma" : 0.95,
              "batch_size" : 1024, 
              "tau" : 0.05,
              "num_layers" : 3, 
              "layer_size" : 512, 
              "verbose" : 0,
              }

# Drama Hyperparameters
drama_params = {}