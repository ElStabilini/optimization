#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_opt_test         # Job name

#run the script
python D1_optimization.py
