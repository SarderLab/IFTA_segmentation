#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/ifta_seg__%j.out
#SBATCH --job-name="new_ifta"


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anish.tatke@ufl.edu.com

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited

date
ml

module load conda
conda activate ifta_tf215

USER=anish.tatke
PROJECT=new_ifta

DATA_DIR=/orange/pinaki.sarder/$USER/IFTA_Seg


python segmentation_school.py \
    --data_dir $DATA_DIR \
    --option predict \
    --project JamieData \
    --encoder_name deeplab \
    --one_network True \
    --batch_size 2 \
    --boxSizeHR 3000 \
    --classNum 4 \
    --overlap_percentHR 0.5 \
    --wsi_ext .tif \