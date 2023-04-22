#!/bin/bash

# Under, Over and Expert Provisioning
python3 train.py -n std -k 0.05 -e 1000
python3 train.py -n op -k 0.03 -e 1000
python3 train.py -n up -k 0.07 -e 1000

# Varying Chain Length and VM Configuration Counts
python3 train.py -n std-f5-c3 -ncf 5 -ncp 3 -e 1000
python3 train.py -n std-f10-c3 -ncf 10 -ncp 3 -e 1000
python3 train.py -n std-f25-c3 -ncf 25 -ncp 3 -e 1000
python3 train.py -n std-f50-c3 -ncf 50 -ncp 3 -e 1000
python3 train.py -n std-f100-c3 -ncf 100 -ncp 3 -e 1000
python3 train.py -n std-f5-c5 -ncf 5 -ncp 5 -e 1000
python3 train.py -n std-f5-c10 -ncf 5 -ncp 10 -e 1000
python3 train.py -n std-f5-c20 -ncf 5 -ncp 20 -e 1000

# hyperparameter evaluation
python3 train.py -n std01 -c 0.1 -k 0.05 -e 500
python3 train.py -n std02 -c 0.2 -k 0.05 -e 500
python3 train.py -n std03 -c 0.3 -k 0.05 -e 500
python3 train.py -n op01 -c 0.1 -k 0.02 -e 500
python3 train.py -n op02 -c 0.2 -k 0.02 -e 500
python3 train.py -n op03 -c 0.3 -k 0.02 -e 500
python3 train.py -n up01 -c 0.1 -k 0.07 -e 500
python3 train.py -n up02 -c 0.2 -k 0.07 -e 500
python3 train.py -n up03 -c 0.3 -k 0.07 -e 500

