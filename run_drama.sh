#!/bin/bash
python experiment.py -a drama -s perpendicular -p results/drama/perpendicular -n 5
python experiment.py -a drama -s diagonal-25 -p results/drama/diagonal-25 -n 5
python experiment.py -a drama -s diagonal-50 -p results/drama/diagonal-50 -n 5
python experiment.py -a drama -s parallel -p results/drama/parallel -n 5
python experiment.py -a drama -s interleave -p results/drama/interleave -n 5
python experiment.py -a drama -s sequential-inc -p results/drama/sequential-inc -n 5
python experiment.py -a drama -s sequential-dec -p results/drama/sequential-dec -n 5