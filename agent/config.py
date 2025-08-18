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
ppo_params = {"gamma" : 0.99,
              "lamb" : 0.9,
              "eps_clip" : 0.1,
              "K_epochs" : 20,
              "num_cells" : 64,
              "actor_lr" : 1e-4,
              "critic_lr" : 3e-4,
              "memory_size" : 2048,
              "minibatch_size" : 64,
              "c1" : .1,
              "c2" : 0.01,
              "kl_threshold" : 0.15,
              "parameters_hardshare" : False,
              "early_stop" : False,
              "cal_total_loss" : True,
              "max_grad_norm" : 0.5,
              "layer_num" : 3
              }

"""
# PPO Hyperparameters tuned parallel-adj
ppo_params = {"gamma" : 0.99, 
              "lamb" : 0.95, 
              "eps_clip" : 0.1, 
              "K_epochs" : 20, 
              "num_cells" : 64, 
              "actor_lr" : 0.0001, 
              "critic_lr" : 0.0001,
              "memory_size" : 4096,
              "minibatch_size" : 128, 
              "c1" : 0.1, 
              "c2" : 0.01, 
              "kl_threshold" : 0.15,
              "parameters_hardshare" : False,
              "early_stop" : False,
              "cal_total_loss" : True,
              "max_grad_norm" : 1,
              "layer_num" : 2,
              }
"""

# SAC Hyperparameters
sac_params = {"buffer_size" : 1_000_000, 
              "learning_rate" : 9e-4, 
              "gamma" : 0.995,
              "batch_size" : 32, 
              "tau" : 0.6,
              "num_layers" : 6, 
              "layer_size" : 1024, 
              "verbose" : 0,
              }

# Drama Hyperparameters
drama_params = {"config" : {
  "BasicSettings": {
    "ImageSize": 128, #64,
    "ImageChannel": 3,
    "ReplayBufferOnGPU": True,
    "Seed": 3710, # Will be overwritten by training seed of the run
    "Env_name": "HW/parking-v0", #ALE/Pong-v5
    #"Device": "cuda:0", # Pass as seperate parameter
    "Use_amp": True,
    "Use_cg": True,
    "Compile": True,
    "SavePath": "None"
  },
  "Evaluate": {
    "EpisodeNum": 10,  # Will be overwritten as n_eval_episodes to ensure consistency with SAC and PPO training params
    "NumEnvs": 10,
    "DuringTraining": True,
    "EverySteps": 1000  # Will be overwritten as eval_freq to ensure consistency with SAC and PPO training params
  },
  "JointTrainAgent": {
    "SampleMaxSteps": 105000,   # Just to make sure the last episode will finish, no training after 100k
                                # Will be overwritten as max_timesteps + 5000 to ensure consistency with SAC and PPO training params
    "BufferMaxLength": 100000,
    "WorldModelWarmUp": 1032,
    "BehaviourWarmUp": 1032,
    "NumEnvs": 1,
    "BatchSize": 16,
    "BatchLength": 128,
    "ImagineBatchSize": 1024,
    "ImagineContextLength": 8,
    "ImagineBatchLength": 16,
    "RealityContextLength": 16,
    "TrainDynamicsEverySteps": 1,
    "TrainDynamicsEpoch": 1,
    "TrainAgentEverySteps": 1,
    "FreezeWorldModelAfterSteps": 100000, # Will be overwritten as max_timesteps to ensure consistency with SAC and PPO training params
    "FreezeBehaviourAfterSteps": 100000,  # Will be overwritten as max_timesteps to ensure consistency with SAC and PPO training params
    "SaveEverySteps": 2000,
    "SaveModels": True,
    "Tau": 10,
    "ImaginationTau": 10,
    "Alpha": 1, # High focus on penalising high imagine counts regardless of train counts, less probability to be sampled
    "Beta": 1 # High focus on penalising 
  },
  "Models": {
    "WorldModel": {
      "dtype": "float32",
      "Backbone": "Mamba2", # Mamba, Mamba2, Transformer
      "InChannels": 3,
      "Act": "SiLU",
      "CategoricalDim": 32,
      "ClassDim": 32,
      "HiddenStateDim": 512,
      "Optimiser": "Laprop",
      "LatentDiscreteType": "naive",
      "Max_grad_norm": 1000,
      "Warmup_steps": 1000,
      "Dropout": 0.1,
      "Unimix_ratio": 0.01,
      "Weight_decay": 0.0001,
      "Adam": {
        "LearningRate": 0.0001
      },
      "Laprop": {
        "LearningRate": 0.00004,
        "Epsilon": 1e-20
      },
      "Encoder": {
        "Depth": 16,
        "Mults": [1, 2, 3, 4, 4],
        "Norm": "rms",
        "Kernel": 5,
        "Padding": "same",
        "InputSize": [3, 64, 64]    # Will be updated to correspond with obervation image size
      },
      "Decoder": {
        "Depth": 16,
        "Mults": [1, 2, 3, 4, 4],
        "Norm": "rms",
        "Kernel": 5,
        "Padding": "same",
        "FirstStrideOne": True,
        "InputSize": [3, 64, 64],   # Will be updated to correspond with obervation image size
        "FinalLayerSigmoid": True
      },
      "Reward": {
        "HiddenUnits": 256,
        "LayerNum": 1
      },
      "Termination": {
        "HiddenUnits": 256,
        "LayerNum": 1
      },
      "Transformer": {
        "FinalFeatureWidth": 4,
        "NumLayers": 2,
        "NumHeads": 8
      },
      "Mamba": {
        "n_layer": 2,
        "d_intermediate": 0,
        "ssm_cfg": {
          "d_state": 16
        }
      }
    },
    "Agent": {
      "dtype": "float32",
      "Policy": "PPO", # AC or PPO
      "Unimix_ratio": 0,
      "AC": {
        "NumLayers": 3,
        "Gamma": 0.985,
        "Lambda": 0.95,
        "EntropyCoef": 0.0003,
        "Max_grad_norm": 100,
        "Warmup_steps": 1000,
        "Act": "SiLU",
        "Optimiser": "Laprop",
        "Adam": {
          "LearningRate": 0.00003,
          "Epsilon": 0.00001
        },
        "Laprop": {
          "LearningRate": 0.00004,
          "Epsilon": 1e-20
        },
        "Actor": {
          "HiddenUnits": 256
        },
        "Critic": {
          "HiddenUnits": 512
        }
      },
      "PPO": {
        "NumLayers": 3,
        "Gamma": 0.985,
        "Lambda": 0.95,
        "EpsilonClip": 0.2,
        "K_epochs": 3,
        "Minibatch": 16384,
        "CriticCoef": 1,
        "EntropyCoef": 0.0003,
        "KL_threshold": 0.01,
        "Max_grad_norm": 100,
        "Warmup_steps": 1000,
        "Act": "SiLU",
        "Optimiser": "Laprop",
        "Adam": {
          "LearningRate": 0.00003,
          "Epsilon": 0.00001
        },
        "Laprop": {
          "LearningRate": 0.00004,
          "Epsilon": 1e-20
        },
        "Actor": {
          "HiddenUnits": 256
        },
        "Critic": {
          "HiddenUnits": 512 # Andrychowicz2020 wider critic network seems better  
        }
      }
    }
  },
  "Wandb": {
    "Init": {
      "Mode": "online",
      "Project": "Mamba_dreamer"
    }
  },
  "n": "standard"
}}
