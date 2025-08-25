#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/cc.out
#SBATCH --job-name="ifta-compat-check"


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anish.tatke@ufl.edu.com

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited

date
ls
ml

module load conda
conda activate ifta_tf215

USER=anish.tatke
PROJECT=ifta-compat-check

python check_gpu_compat.py