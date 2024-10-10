#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_opt         # Job name

#run the script
python rb_optimization.py
