#!/bin/bash
#SBATCH --job-name="edm"
#SBATCH --partition=h200
#SBATCH --nodelist=hci-06
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # DO NOT CHANGE, allocate all gpus on single task per node (for slurm), so torchrun uses as many processes per node as gpus
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%j.out

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
echo "Running on host:: $RDZV_HOST"

# for ffhq128_grayscale, following ffhq64 config in EDM paper except enlarging --cbase 192 --cres 1,2,2,2,2 \
# takes 69G on each gpu with --batch-gpu 64
# parameter count: 183M
srun torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc-per-node 4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    train.py \
    --outdir training-runs \
    --data-name ffhq128 --data /home/share/normal_share/ffhq256 \
    --resize 128 --grayscale True --max-images 68000 --workers 2 \
    --arch ncsnpp --cbase 192 --cres 1,2,2,2,2 \
    --lr 2e-4 --batch 256 --batch-gpu 64 --fp16 True \
    --dropout 0.05 --augment 0.15 --xflip True \
    --tick 50 --snap 50 --dump 500