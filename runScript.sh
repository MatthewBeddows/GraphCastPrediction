#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_full
#SBATCH --nodelist=agpu004
 
# module load miniconda3
source /opt/software/uoa/apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate graphcast
 
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
 
 
srun python GraphCastDemoPythonVersion.py 
 
