#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --output=trappm-out.%j
#SBATCH --error=trappmr-err.%j
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

ml Stages/2023
ml Python
ml CUDA

source venv/bin/activate
cd source-code/
srun python main.py
