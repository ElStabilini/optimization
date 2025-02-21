#!/bin/bash

#SBATCH --partition=qw11q 
#SBATCH --job-name=rb_NM         # Job name

#run the python script
python init_simplex.py
