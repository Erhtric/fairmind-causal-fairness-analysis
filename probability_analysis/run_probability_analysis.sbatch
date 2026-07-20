#!/bin/bash
#SBATCH --job-name=probability_analysis
# partition=gpu (NIENTE --gres=gpu:1: non serve una GPU, solo CPU/mem).
# Su "compute" la risoluzione utente NSS/LDAP e' rotta dentro i job batch
# (vedi notebooks/soldani_task/setup_thor/COME_LANCIARE_NOTEBOOK.txt).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=probability_analysis_%j.out
#SBATCH --error=probability_analysis_%j.err

SCRATCH=/mnt/beegfs/scratch/marco.soldani/causal_fairness
JUPYTER_SIF=$SCRATCH/containers/jupyter_env.sif
PORT=8080
NOTEBOOK_DIR=/home/marco.soldani/causal_fairness/probability_analysis

if [ ! -f "$JUPYTER_SIF" ]; then
    echo "ERRORE: $JUPYTER_SIF non esiste. Costruiscilo prima (vedi setup_thor/)."
    exit 1
fi

echo "=== Cerco un job llama_server_gpu gia' in esecuzione ==="
LLAMA_HOST=$(squeue --name=llama_server_gpu --states=RUNNING -h -o "%N" | head -1)

if [ -z "$LLAMA_HOST" ]; then
    echo "ERRORE: nessun job 'llama_server_gpu' in stato RUNNING trovato."
    echo "Avvia prima il server persistente con: sbatch notebooks/soldani_task/setup_thor/llama_server.sbatch"
    exit 1
fi

echo "llama-server trovato su $LLAMA_HOST:$PORT"
export LLAMA_HOST
export LLAMA_PORT=$PORT

echo "=== Esecuzione notebook probability_analysis ==="
cd $NOTEBOOK_DIR
apptainer exec --no-home \
    -B $SCRATCH:/scratch \
    -B /home/marco.soldani:/home/marco.soldani \
    --env LLAMA_HOST=$LLAMA_HOST \
    --env LLAMA_PORT=$PORT \
    $JUPYTER_SIF \
    jupyter nbconvert --to notebook --execute \
        --output analyze_llm_probability_interpretation_output_${SLURM_JOB_ID}.ipynb \
        analyze_llm_probability_interpretation.ipynb

if [ $? -ne 0 ]; then
    echo "ERRORE: esecuzione del notebook fallita (vedi output sopra)."
    exit 1
fi

echo "=== Analisi completata ==="
echo "Output: $NOTEBOOK_DIR/analyze_llm_probability_interpretation_output_${SLURM_JOB_ID}.ipynb"
echo "Risultati JSON: $NOTEBOOK_DIR/results/"
