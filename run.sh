#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000mb
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --time=168:00:00
#SBATCH --output=hail.out
#SBATCH --job-name="PTCs"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module list
which python

echo "Launch job"
CUDA_LAUNCH_BLOCKING=1
#python3 test.py
python3 generate_output_unet_cmd.py --patchsize=512 --batchsize=1 --outdir=/blue/pinaki.sarder/nlucarelli/PTC/KPMP_ptcs/ --model=/blue/pinaki.sarder/nlucarelli/PTC/pas-peritubular-capillaries-40X_unet_best_model.pth /orange/pinaki.sarder/nlucarelli/KPMP/*.svs


echo "All Done!"
