#!/bin/bash
#SBATCH --job-name=fairmind_benchmark
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# === CONFIGURAZIONE ===
SCRATCH=/mnt/beegfs/scratch/marco.soldani/causal_fairness
JUPYTER_SIF=$SCRATCH/containers/jupyter_env.sif
PORT=8080
NOTEBOOK_DIR=/home/marco.soldani/causal_fairness/notebooks/soldani_task

if [ ! -f "$JUPYTER_SIF" ]; then
    echo "ERRORE: $JUPYTER_SIF non esiste ancora."
    echo "Costruiscilo una tantum (il .def e' nel repo git, in home) con:"
    echo "  apptainer build $JUPYTER_SIF /home/marco.soldani/causal_fairness/notebooks/soldani_task/setup_thor/jupyter_env.def"
    exit 1
fi

# Questo job NON avvia un llama-server proprio: riusa quello persistente
# gia' lanciato con `sbatch llama_server.sbatch` (job-name=llama_server_gpu,
# vedi slurm/llama_server_gpu.sbatch su scratch). Evita di dover chiedere
# una seconda GPU, che manderebbe il job in PENDING a tempo indefinito.
echo "=== Cerco un job llama_server_gpu gia' in esecuzione ==="
# Niente "-u $USER": dentro un job batch la risoluzione dell'utente via
# NSS/LDAP a volte fallisce ("squeue: error: Invalid user: ...") anche
# se l'utente e' valido. Il nome del job basta a identificarlo.
LLAMA_HOST=$(squeue --name=llama_server_gpu --states=RUNNING -h -o "%N" | head -1)

if [ -z "$LLAMA_HOST" ]; then
    echo "ERRORE: nessun job 'llama_server_gpu' in stato RUNNING trovato."
    echo "Avvia prima il server persistente con: sbatch llama_server.sbatch"
    exit 1
fi

echo "llama-server trovato su $LLAMA_HOST:$PORT"
export LLAMA_HOST
export LLAMA_PORT=$PORT

echo "=== Esecuzione notebook benchmark ==="
cd $NOTEBOOK_DIR
# --no-home: evita che apptainer provi a risolvere l'utente host per
# bindare automaticamente la home (fallisce con "Couldn't determine
# user account information" per lo stesso motivo NSS/LDAP di squeue).
# La home ci serve comunque gia' bindata esplicitamente sotto.
apptainer exec --no-home \
    -B $SCRATCH:/scratch \
    -B /home/marco.soldani:/home/marco.soldani \
    --env LLAMA_HOST=$LLAMA_HOST \
    --env LLAMA_PORT=$PORT \
    $JUPYTER_SIF \
    jupyter nbconvert --to notebook --execute \
        --output benchmark_output_${SLURM_JOB_ID}.ipynb \
        2_3_benchmark_thor.ipynb

if [ $? -ne 0 ]; then
    echo "ERRORE: esecuzione del notebook fallita (vedi output sopra)."
    exit 1
fi

echo "=== Benchmark completato ==="
echo "Output: $NOTEBOOK_DIR/benchmark_output_${SLURM_JOB_ID}.ipynb"
