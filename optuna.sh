#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_optuna         # Job name

#run the script
python D1_optuna.py

