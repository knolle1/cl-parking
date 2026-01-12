#!/bin/bash
python experiment.py -a sac -s perpendicular -p results/sac/perpendicular -d "SAC trained in perpendicular parking scenario" -n 5 --gpu 0 &
python experiment.py -a sac -s diagonal-25 -p results/sac/diagonal-25 -d "SAC trained in 25 degree diagonal parking scenario" -n 5 --gpu 0 &
python experiment.py -a sac -s diagonal-50 -p results/sac/diagonal-50 -d "SAC trained in 50 degree diagonal parking scenario" -n 5 --gpu 0 &
python experiment.py -a sac -s parallel -p results/sac/parallel -d "SAC trained in parallel parking scenario" -n 5 --gpu 0 &
python experiment.py -a sac -s interleave -p results/sac/interleave -d "SAC trained in interleaved parking scenarios" -n 5 --gpu 0  &
python experiment.py -a sac -s sequential-inc -p results/sac/sequential-inc -d "SAC trained in sequential parking scenarios (perpendicular -> 25 degree -> 50 degree -> parallel)" -n 5 --gpu 0
python experiment.py -a sac -s sequential-dec -p results/sac/sequential-dec -d "SAC trained in sequential parking scenarios (parallel -> 50 degree -> 25 degree -> perpendicular)" -n 5 --gpu 0