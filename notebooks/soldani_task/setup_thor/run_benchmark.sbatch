#!/bin/bash
#SBATCH --job-name=fairmind_benchmark
#SBATCH --partition=gpuISIN
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# === CONFIGURAZIONE ===
SCRATCH=/mnt/beegfs/scratch/marco.soldani/causal_fairness
SERVER_SIF=$SCRATCH/llama_server.sif
JUPYTER_SIF=$SCRATCH/jupyter_env.sif
MODEL=$SCRATCH/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
PORT=8080
NOTEBOOK_DIR=/home/marco.soldani/causal_fairness/notebooks/soldani_task
LLAMA_SERVER_PID=0

cleanup() {
    echo "=== Cleanup: fermo llama-server (PID=$LLAMA_SERVER_PID) ==="
    if [ $LLAMA_SERVER_PID -ne 0 ]; then
        kill $LLAMA_SERVER_PID 2>/dev/null || true
        wait $LLAMA_SERVER_PID 2>/dev/null || true
    fi
    echo "=== Cleanup completato ==="
}
trap cleanup EXIT

echo "=== Avvio llama-server in background ==="
srun apptainer exec --nv \
    -B $SCRATCH:/scratch \
    $SERVER_SIF \
    llama-server \
        --model /scratch/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --host 0.0.0.0 \
        --port $PORT \
        --ctx-size 32768 \
        --n-predict 1024 \
        --threads 8 \
        --n-gpu-layers 999 &
LLAMA_SERVER_PID=$!

NODE=$(hostname)
echo "llama-server in avvio su $NODE:$PORT (PID=$LLAMA_SERVER_PID)"

echo "=== Attendo che llama-server sia pronto ==="
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "llama-server pronto dopo ${i}s"
        break
    fi
    if ! kill -0 $LLAMA_SERVER_PID 2>/dev/null; then
        echo "ERRORE: llama-server si e' arrestato durante l'avvio"
        exit 1
    fi
    sleep 2
done

export LLAMA_HOST=$NODE
export LLAMA_PORT=$PORT
export LLAMA_ENDPOINT="http://${NODE}:${PORT}/v1"

echo "=== Esecuzione notebook benchmark ==="
cd $NOTEBOOK_DIR
apptainer exec \
    -B $SCRATCH:/scratch \
    -B /home/marco.soldani:/home/marco.soldani \
    --env LLAMA_HOST=$NODE \
    --env LLAMA_PORT=$PORT \
    $JUPYTER_SIF \
    jupyter nbconvert --to notebook --execute \
        --output benchmark_output_${SLURM_JOB_ID}.ipynb \
        2_1_benchmark_thor.ipynb

echo "=== Benchmark completato ==="
echo "Output: $NOTEBOOK_DIR/benchmark_output_${SLURM_JOB_ID}.ipynb"
