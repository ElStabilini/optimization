#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_opt_cma         # Job name

#run the script
python D1_cma.py
