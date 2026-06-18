#!/bin/bash
#SBATCH -A 
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --gpus 1
#SBATCH -t 9:00:00

echo "Running synth training on..."

python run_gen.py

# Deactivate environment after execution
mamba deactivate
