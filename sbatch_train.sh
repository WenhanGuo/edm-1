#!/bin/bash
#SBATCH --job-name="edm"
#SBATCH --partition=defq
#SBATCH --nodelist=hci-05
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # DO NOT CHANGE, allocate all gpus on single task per node (for slurm), so torchrun uses as many processes per node as gpus
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
echo "Running on host:: $RDZV_HOST"

# for ffhq128_grayscale, following ffhq64 config in EDM paper, changed model_channels to 192
# takes 68G on each of four RTX6000 with --batch-gpu 16
# parameter count: 138M
srun torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc-per-node 4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    train.py \
    --outdir training-runs \
    --data /home/share/normal_share/ffhq256 --max-images 68000 --grayscale True \
    --arch ncsnpp --cbase 192 --cres 1,2,2,2 \
    --lr 2e-4 --batch 128 --batch-gpu 16 --fp16 True \
    --dropout 0.05 --augment 0.15 --xflip True \
    --tick 50 --snap 50 --dump 500