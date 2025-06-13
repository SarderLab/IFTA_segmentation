#!/bin/sh
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --time=72:00:00
#SBATCH --output=./slurm_log.out
#SBATCH --job-name="ifta:t0"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
# module load singularity
# module load pytorch
# ls
# ml

# USER=sdevarasetty
PROJECT=TxR01

CODESDIR=.

SIFDIR=./singularity

DATADIR=$CODESDIR/test_data
MODELDIR=./$PROJECT/MODELS/

CONTAINER=$SIFDIR/ifta2.sif
CUDA_LAUNCH_BLOCKING=1

python3 segmentation_school.py --option predict --base_dir $CODESDIR --project $PROJECT --encoder_name deeplab --one_network True --classNum 4 --boxSizeHR 3000 --overlap_percentHR 0.5
