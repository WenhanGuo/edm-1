#!/bin/bash
#SBATCH --job-name="edm"
#SBATCH --partition=defq
#SBATCH --nodelist=hci-02
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%j.out


# For FFHQ and AFHQv2 at 64x64, use deterministic sampling with 40 steps (NFE = 79)
# doubled NFE for FFHQ 128
srun python generate.py \
    --outdir samples_ffhq256 \
    --steps 80 \
    --seeds 0-7 --batch 8 \
    --network training-runs/00003-ffhq256-uncond-ncsnpp-edm-gpus4-batch128-fp16/network-snapshot-007507.pkl