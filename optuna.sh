#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --time=07:00:00            # Time limit
#SBATCH --job-name=rb_optuna         # Job name

#run the script
python D1_optuna.py

