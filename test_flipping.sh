#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_optuna         # Job name

#run the script
python sequence_test.py

