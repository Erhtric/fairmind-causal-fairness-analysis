#!/bin/bash
#SBATCH --job-name=llama_server
#SBATCH --partition=gpuISIN
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=llama_server_%j.out
#SBATCH --error=llama_server_%j.err

SCRATCH=/mnt/beegfs/scratch/marco.soldani/causal_fairness
SIF=$SCRATCH/llama_server.sif
MODEL=$SCRATCH/models/qwen2.5-7b-instruct-q4_k_m.gguf
PORT=8080

NODE=$(hostname)
echo "Job in esecuzione su: $NODE"
echo "Endpoint: http://$NODE:$PORT/v1"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

srun apptainer exec --nv \
    -B $SCRATCH:/scratch \
    $SIF \
    llama-server \
        --model /scratch/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --host 0.0.0.0 \
        --port $PORT \
        --ctx-size 131072 \
        --n-predict 1024 \
        --threads 8 \
        --n-gpu-layers 999
