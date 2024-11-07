#!/bin/bash
#SBATCH -J patch_opt
#SBATCH -c 2
#SBATCH -p seas_gpu,gpu
#SBATCH -t 0-10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --constraint="a100|v100|a40"
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=liviaguttieres@g.harvard.edu 
#SBATCH -o /n/brenner_lab/Lab/rotation_livia/self-assembly-toolkit/dimers/results/patch_opt.out
#SBATCH -e /n/brenner_lab/Lab/rotation_livia/self-assembly-toolkit/dimers/results/patch_opt.err

module load cuda/12.2.0-fasrc01

# Activate conda environment
source "/n/home02/lguttieres/miniconda3/etc/profile.d/conda.sh"
export PATH="/n/home02/lguttieres/miniconda3/condabin:${PATH}"
conda activate jax

# Change to the directory where the bash script is located
cd /n/brenner_lab/Lab/rotation_livia/self-assembly-toolkit/optimization/dimers

# Create results directory if it doesn't exist
mkdir -p /n/brenner_lab/Lab/rotation_livia/self-assembly-toolkit/optimization/dimers

# Run the Python script
python3 computepatch.py

# Notify when the job is done
echo "Job completed"

