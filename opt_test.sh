#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_opt         # Job name
#SBATCH --time=03:00:00            # Time limit

#run the script
python rb_optimization.py
