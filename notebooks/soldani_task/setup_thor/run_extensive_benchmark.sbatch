#!/bin/bash
#SBATCH --job-name=fairmind_extensive_benchmark
# partition=gpu (NIENTE --gres=gpu:1: non ci serve una GPU, solo CPU/mem).
# Su "compute" la risoluzione utente NSS/LDAP dei nodi e' rotta dentro i
# job batch (whoami/apptainer falliscono con "unknown userid"), confermato
# il 20/07/2026. Su "gpu" funziona (stesso partition del server LLM).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=extensive_benchmark_%j.out
#SBATCH --error=extensive_benchmark_%j.err

# set -u: un nome di variabile sbagliato diventa un errore invece di espandersi
#   a stringa vuota (senza, un tipo in ${SLURM_JOB_ID} produrrebbe un file di
#   output chiamato "..._.ipynb" senza che nessuno se ne accorga).
# set -o pipefail: lo stato di una pipeline diventa quello del primo comando
#   fallito, non quello dell'ultimo.
#
# NIENTE set -e, deliberatamente: questi script controllano gli exit code a
# mano per stampare messaggi che dicono a quale passo tornare. Con -e il job
# terminerebbe prima di stamparli, perdendo l'unica diagnostica utile.
set -uo pipefail


# Esegue 2_8_extensive_benchmark.ipynb: gli stessi due percorsi di 2_7,
# ripetuti sui dieci dataset configurati in src/dataset_configs.py.
#
# --time=06:00:00: ogni dataset fa TRE richieste al modello, e quella centrale
# e' la lunga, perche' enumera una riga per ogni coppia (z,w). Su Adult, con 25
# coppie, sono poco piu' di due minuti; i dataset piu' grossi ne hanno 30. Una
# stima prudente e' mezz'ora di inferenza piu' il fitting delle reti, che su
# census_income_kdd legge duecentomila righe. Sei ore lasciano margine anche se
# il server e' conteso.
#
# Il notebook NON si ferma al primo dataset che fallisce: run_one() cattura
# l'eccezione, la registra nel record con il traceback e prosegue. Un dataset
# perso non deve costare gli altri nove. Il conteggio finale dice quanti sono
# arrivati in fondo, e il job termina comunque con errore se il notebook stesso
# si rompe.

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

# src/benchmark_common.py e' nuovo rispetto ai job precedenti: se il repo su
# Thor non e' aggiornato il notebook fallisce sull'import, dopo aver gia'
# occupato il nodo. Meglio accorgersene qui.
COMMON=/home/marco.soldani/causal_fairness/src/benchmark_common.py
CONFIGS=/home/marco.soldani/causal_fairness/src/dataset_configs.py
for f in "$COMMON" "$CONFIGS"; do
    if [ ! -f "$f" ]; then
        echo "ERRORE: $f non esiste."
        echo "Il repo su Thor non e' allineato. Sincronizza src/ dal Mac con:"
        echo "  rsync -av --exclude='__pycache__' src/ marco.soldani@thor.supsi.ch:/home/marco.soldani/causal_fairness/src/"
        exit 1
    fi
done

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

echo "=== Esecuzione notebook complete pipeline (due percorsi) ==="
cd $NOTEBOOK_DIR
# --no-home: evita che apptainer provi a risolvere l'utente host per
# bindare automaticamente la home (fallisce con "Couldn't determine
# user account information" per lo stesso motivo NSS/LDAP di squeue).
# La home ci serve comunque gia' bindata esplicitamente sotto.
#
# NIENTE --allow-errors: se una delle tre chiamate al modello fallisce, o se
# il modello non rispetta il template, vogliamo che il job termini con errore
# invece di produrre un notebook di output con una cella rossa che passa
# inosservata. Vale a maggior ragione qui: un ramo dei due potrebbe fallire
# e l'altro no, e un risultato a meta' sarebbe peggio di nessun risultato.
apptainer exec --no-home \
    -B $SCRATCH:/scratch \
    -B /home/marco.soldani:/home/marco.soldani \
    --env LLAMA_HOST=$LLAMA_HOST \
    --env LLAMA_PORT=$PORT \
    $JUPYTER_SIF \
    jupyter nbconvert --to notebook --execute \
        --output extensive_benchmark_output_${SLURM_JOB_ID}.ipynb \
        2_8_extensive_benchmark.ipynb
status=$?

if [ $status -ne 0 ]; then
    echo "ERRORE: esecuzione del notebook fallita (vedi output sopra)."
    echo "Se l'errore e' 'The model did not finish its answer', la risposta e'"
    echo "stata troncata: e' il guard su finish_reason, non un bug. Alza"
    echo "max_tokens nella cella corrispondente e rilancia."
    exit 1
fi

echo "=== Complete pipeline completata ==="
echo "Notebook eseguito: $NOTEBOOK_DIR/extensive_benchmark_output_${SLURM_JOB_ID}.ipynb"
echo "Report LaTeX + JSON: $NOTEBOOK_DIR/benchmark_results/extensive/<timestamp>/"
echo
echo "Nella cartella trovi summary.csv con una riga per dataset, e"
echo "decomposition.csv con la decomposizione di riferimento di ciascuno."
echo "summary.json contiene i record completi, inclusi i traceback dei"
echo "dataset eventualmente falliti."
echo
# pdflatex non e' installato nel container: la verifica di compilazione va
# fatta in locale sui .tex scaricati.
echo "NOTA: la compilazione LaTeX non viene verificata su Thor (pdflatex"
echo "      non e' nel container). Per verificarla, scarica i .tex e compila"
echo "      in locale con:"
echo "        pdflatex -interaction=nonstopmode <file>.tex"
