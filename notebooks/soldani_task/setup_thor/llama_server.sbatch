#!/bin/bash
#SBATCH --job-name=llama_server_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --output=llama_server_gpu_%j.out
#SBATCH --error=llama_server_gpu_%j.err

export SCRATCH=/mnt/beegfs/scratch/marco.soldani/causal_fairness
export SIF=$SCRATCH/containers/llama_server_gpu.sif
export PORT=8080

srun apptainer exec --nv \
    -B $SCRATCH:/scratch \
    $SIF \
    /usr/local/bin/llama-server \
        --model /scratch/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
        --host 0.0.0.0 \
        --port $PORT \
        --ctx-size 131072 \
        --n-gpu-layers 999 \
        --n-predict 1024 \
        --threads 4
