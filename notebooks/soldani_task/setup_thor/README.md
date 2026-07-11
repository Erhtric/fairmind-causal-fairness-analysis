# FairMind Benchmark su Thor

Due approcci: **automatico** (singolo sbatch) o **interattivo** (server persistente + Jupyter via tunnel SSH).

## 0. Setup iniziale (una tantum)

```bash
ssh name.surname@thor.supsi.ch

# Su hnode01 (login node)
mkdir -p /mnt/beegfs/scratch/marco.soldani/causal_fairness/models
cd /mnt/beegfs/scratch/marco.soldani/causal_fairness

# Build container GPU per il server LLM
apptainer build llama_server.sif setup_thor/llama_server.def

# Build container CPU per Jupyter + FairMind
apptainer build jupyter_env.sif setup_thor/jupyter_env.def

# Scarica modello GGUF (se non già presente)
cd models
curl -LO https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
```

## 1. Flusso automatico (singolo job)

Un solo sbatch: avvia il server, esegue il notebook, ferma tutto.

```bash
cd /home/marco.soldani/causal_fairness/notebooks/soldani_task
sbatch setup_thor/run_benchmark.sbatch
tail -f benchmark_<jobid>.out
```

Output salvato in `benchmark_output_<jobid>.ipynb`.

## 2. Flusso interattivo (due job + tunnel SSH)

### 2a. Avvia il server LLM (job persistente)

```bash
cd /home/marco.soldani/causal_fairness/notebooks/soldani_task
sbatch setup_thor/llama_server.sbatch
squeue -u $USER          # prendi nota del JOBID

# Leggi il nodo assegnato (es. gnode05) dal file di output
tail -f llama_server_<jobid>.out
# Cerca: "Job in esecuzione su: gnode05"
# Cerca: "Endpoint: http://gnode05:8080/v1"
```

Il modello impiega ~1-2 minuti per caricarsi in GPU.

### 2b. Avvia Jupyter (job interattivo separato)

```bash
srun --partition=compute --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
```

Dentro al nodo compute (es. `cnode02`):

```bash
cd /mnt/beegfs/scratch/marco.soldani/causal_fairness
apptainer exec jupyter_env.sif \
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Annota l'hostname del nodo compute (es. `cnode02`) e il token Jupyter.

### 2c. Tunnel SSH dal Mac

In un **nuovo terminale locale** (non su Thor):

```bash
ssh -L 8888:cnode02:8888 marcosoldani@thor.supsi.ch
```

Ora apri `http://localhost:8888` nel browser, incolla il token.

**Nota:** il nodo compute e il nodo GPU si vedono sulla rete interna del cluster, quindi dal notebook puoi chiamare `http://gnode05:8080/v1` direttamente.

### 2d. Modifica il notebook

Nel notebook `2_1_benchmark_thor.ipynb`, sostituisci la cella di connessione con il codice in `setup_thor/notebook_patch.py`.

Imposta le variabili in cima:
```python
LLAMA_HOST = "gnode05"   # dal punto 2a
LLAMA_PORT = 8080
```

Poi esegui tutte le celle.

### 2e. Fine sessione

```bash
exit                         # esce da srun interattivo (ferma Jupyter)
scancel <JOBID_llama_server> # ferma il server LLM
```

## 3. Benchmark su più dataset

Modifica `CONFIG` nel notebook e rilanci. Per automatizzare,
usa `run_benchmark.sbatch` con una configurazione esterna.

## Troubleshooting

| Problema | Causa | Fix |
|---|---|---|
| `llama-server si e' arrestato` | GPU occupata o assente | `squeue -u $USER`, verifica conflitti |
| Output notebook tutti zeri | LLM non valido | Controlla il modello in `$SCRATCH/models/` |
| Jupyter non risponde | Tunnel sbagliato o nodo errato | Verifica hostname: `ssh -L 8888:hostname:8888 ...` |
| `apptainer` build fallisce | Cache o permessi | Prova con `--remote` o contatta sysadmin |
| I nodi non si vedono in rete | Firewall interno | Usa il flusso automatico (tutto sullo stesso nodo) |

## Alternative

Per modelli diversi da Qwen 2.5 (es. Llama 3, Mixtral) cambia il file `.gguf` e
la variabile `MODEL_NAME`. Se vuoi usare **vLLM** invece di llama.cpp, crea il
file `vllm_server.def` e lo sbatch corrispondente — l'architettura dei flussi
resta identica.
