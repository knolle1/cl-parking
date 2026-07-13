# Assessing Robustness to Catastrophic Forgetting in Continual Reinforcement Learning for Cyber-Physical Systems

This project evaluates the behaviour of RL algorithms in continual learning settings.

Parking environment adapted from [HighwayEnv](https://github.com/eleurent/highway-env) (date accessed: 12/05/2024)

PPO implementation adapted from implementation by [Wenlong Wang](https://github.com/realwenlongwang/PPO-Single-File-Notebook-Implementation) (date accessed: 25/06/2024)

## Abstract

As the use of reinforcement learning (RL) becomes more widespread in cyber-physical systems, it becomes increasingly important to design RL agents that are able to adapt to changes in their environments in order to maintain safety and reliability. This paper addresses the question of whether different classes of RL algorithm are more robust against forgetting previously learned behaviours, a critical issue in continual learning, in non-stationary environments. We evaluate the robustness of representative examples of three important classes of RL algorithm in a non-stationary autonomous parking environment. These are Proximal Policy Optimization (PPO), which is a _model-free on-policy_ algorithm, Soft-Actor-Critic (SAC), which is a _model-free off-policy_ algorithm, and Drama, which is a _model-based_ algorithm. The agents must learn to successfully park in four different scenarios, which are presented successively, with parking spaces oriented at varying angles. We find off-policy algorithms appear to be more robust compared to on-policy algorithms due to their use of replay, while model-based algorithms do not appear to offer any direct advantage over model-free algorithms. These experiments also highlight the additional challenges of finding robust abstractions of the environment, and oversensitivity to hyperparameters.

## Dependencies

1. Create and activate virtual environment:
	- With conda:
		- `conda create -n cl-parking python=3.10`
		- `conda activate cl-parking`
	- Alternatively with venv (requires python 3.10):
		- `python -m venv .venv`
		- `source .venv/bin/activate`
		
2. Install tools needed to build and install dependencies:
	- `pip install --upgrade pip`
	- `pip install packaging`
	- `pip install setuptools==69.5.1 wheel`
        
3. Install dependencies in the following order:
	- `pip install numpy==1.24.3`
	- `pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121`
	- `pip install causal-conv1d==1.5.0.post8 --no-build-isolation`
	- `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`
	- `pip install -r requirements.txt`
