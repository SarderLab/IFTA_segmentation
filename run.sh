#!/bin/sh
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --time=72:00:00
#SBATCH --output=./slurm_log.out
#SBATCH --job-name="ifta:t0"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load singularity
module load pytorch
ls
ml

USER=sdevarasetty

singularity exec --nv -B $(pwd):/exec/, IFTA.sif python3 /exec/segmentation_school.py --option predict --project TxR01 --encoder_name deeplab --one_network True --classNum 4 --boxSizeHR 3000 --overlap_percentHR 0.5