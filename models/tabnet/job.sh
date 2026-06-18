#!/bin/bash
#SBATCH -A 
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --gpus 1
#SBATCH -t 6:00:00

echo "Running model..."
python run_tn.py  

# Deactivate environment after execution
