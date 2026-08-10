#!/bin/bash
#SBATCH --job-name=fairmind_report_pipeline
# partition=gpu (NIENTE --gres=gpu:1: non ci serve una GPU, solo CPU/mem).
# Su "compute" la risoluzione utente NSS/LDAP dei nodi e' rotta dentro i
# job batch (whoami/apptainer falliscono con "unknown userid"), confermato
# il 20/07/2026. Su "gpu" funziona (stesso partition del server LLM).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=report_pipeline_%j.out
#SBATCH --error=report_pipeline_%j.err

# Esegue 2_6_report_pipeline.ipynb: FairMind calcola i cinque effetti, l'LLM
# riceve i numeri gia' pronti + il template LaTeX rigido e scrive solo la
# parte qualitativa, poi lo scorer valuta le Recap Questions.
#
# Rispetto a run_benchmark.sbatch la richiesta all'LLM e' una sola e molto
# piu' corta (nessun calcolo passo-passo su 25 combinazioni (z,w)), da qui
# --time=01:00:00 invece di 02:00:00.
#
# Il container jupyter_env.sif attuale basta cosi' com'e': il modulo
# src/report_pipeline/ usa solo la stdlib piu' openai, gia' presente
# (openai==2.26.0 in jupyter_env.def). Nessun rebuild necessario.

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
# gia' lanciato con `sbatch llama_server.sbatch` (job-name=llama_server_gpu).
# Evita di dover chiedere una seconda GPU, che manderebbe il job in PENDING
# a tempo indefinito.
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

echo "=== Esecuzione notebook report pipeline ==="
cd $NOTEBOOK_DIR
# --no-home: evita che apptainer provi a risolvere l'utente host per
# bindare automaticamente la home (fallisce con "Couldn't determine
# user account information" per lo stesso motivo NSS/LDAP di squeue).
# La home ci serve comunque gia' bindata esplicitamente sotto.
#
# NIENTE --allow-errors: se la chiamata all'LLM fallisce o il modello non
# rispetta il template, vogliamo che il job termini con errore invece di
# produrre un notebook di output con una cella rossa che passa inosservata.
apptainer exec --no-home \
    -B $SCRATCH:/scratch \
    -B /home/marco.soldani:/home/marco.soldani \
    --env LLAMA_HOST=$LLAMA_HOST \
    --env LLAMA_PORT=$PORT \
    $JUPYTER_SIF \
    jupyter nbconvert --to notebook --execute \
        --output report_pipeline_output_${SLURM_JOB_ID}.ipynb \
        2_6_report_pipeline.ipynb

if [ $? -ne 0 ]; then
    echo "ERRORE: esecuzione del notebook fallita (vedi output sopra)."
    exit 1
fi

echo "=== Report pipeline completata ==="
echo "Notebook eseguito: $NOTEBOOK_DIR/report_pipeline_output_${SLURM_JOB_ID}.ipynb"
echo "Report LaTeX + scoring JSON: $NOTEBOOK_DIR/benchmark_results/reports/"
echo
# pdflatex non e' installato nel container: la cella di verifica lo rileva
# con shutil.which() e salta il controllo senza fallire (nel JSON finale
# "latex_compiles" resta null). Per verificare che il .tex compili davvero,
# scaricalo e compilalo in locale:
echo "NOTA: la compilazione LaTeX non viene verificata su Thor (pdflatex"
echo "      non e' nel container): nel JSON \"latex_compiles\" sara' null."
echo "      Per verificarla, scarica il .tex e compilalo in locale con:"
echo "        pdflatex -interaction=nonstopmode <file>.tex"
