#!/bin/bash
#python experiment.py -a sac -s perpendicular -p results/sac/perpendicular -n 0 --gpu 0
#python experiment.py -a sac -s diagonal-25 -p results/sac/diagonal-25 -n 0 --gpu 0
#python experiment.py -a sac -s diagonal-50 -p results/sac/diagonal-50 -n 0 --gpu 0
#python experiment.py -a sac -s parallel -p results/sac/parallel -n 0 --gpu 0
#python experiment.py -a sac -s interleave -p results/sac/interleave -n 5 --gpu 0 --start 4
python experiment.py -a sac -s sequential-inc -p results/sac/sequential-inc -n 5 --gpu 1 --start 2
python experiment.py -a sac -s sequential-dec -p results/sac/sequential-dec -n 5 --gpu 1