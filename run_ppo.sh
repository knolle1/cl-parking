#!/bin/bash
python experiment.py -a ppo -s perpendicular -p results/ppo/perpendicular -d "PPO trained in perpendicular parking scenario" -n 5 --gpu 0 &
python experiment.py -a ppo -s diagonal-25 -p results/ppo/diagonal-25 -d "PPO trained in 25 degree diagonal parking scenario" -n 5 --gpu 0 &
python experiment.py -a ppo -s diagonal-50 -p results/ppo/diagonal-50 -d "PPO trained in 50 degree diagonal parking scenario" -n 5 --gpu 0 &
python experiment.py -a ppo -s parallel -p results/ppo/parallel -d "PPO trained in parallel parking scenario" -n 5 --gpu 0 &
python experiment.py -a ppo -s parallel-adj -p results/ppo/parallel-adj -d "PPO trained in parallel parking scenario with adjusted reward function" -n 5 --gpu 0 &
python experiment.py -a ppo -s interleave -p results/ppo/interleave -d "PPO trained in interleaved parking scenario" -n 5 --gpu 0 &
python experiment.py -a ppo -s sequential-inc -p results/ppo/sequential-inc -d "PPO trained in sequential parking scenarios (perpendicular -> 25 degree -> 50 degree -> parallel)" -n 5 --gpu 0 &
python experiment.py -a ppo -s sequential-dec -p results/ppo/sequential-dec -d "PPO trained in sequential parking scenarios (parallel -> 50 degree -> 25 degree -> perpendicular)" -n 5 --gpu 0