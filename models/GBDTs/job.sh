#!/bin/bash
#SBATCH -A Berzelius-2025-10
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=berzelius-cpu
#SBATCH -t 48:00:00


echo "Running model..."
python main.py  

# Deactivate environment after execution
mamba deactivate 
