#!/bin/bash
#SBATCH -A Berzelius-2025-10
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --gpus 1
#SBATCH -t 9:00:00


echo "Running synth training on..."

python main.py --tvae 

# Deactivate environment after execution

