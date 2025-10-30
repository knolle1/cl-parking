#!/bin/bash
python experiment.py -a ppo -s perpendicular -p results/ppo/perpendicular -n 0 --gpu 0
python experiment.py -a ppo -s diagonal-25 -p results/ppo/diagonal-25 -n 0 --gpu 0
python experiment.py -a ppo -s diagonal-50 -p results/ppo/diagonal-50 -n 0 --gpu 0
python experiment.py -a ppo -s parallel -p results/ppo/parallel -n 0 --gpu 0
python experiment.py -a ppo -s parallel-adj -p results/ppo/parallel-adj -n 0 --gpu 0
python experiment.py -a ppo -s interleave -p results/ppo/interleave -n 0 --gpu 0
python experiment.py -a ppo -s sequential-inc -p results/ppo/sequential-inc -n 5 --gpu 0 --start 1
python experiment.py -a ppo -s sequential-dec -p results/ppo/sequential-dec -n 5 --gpu 0