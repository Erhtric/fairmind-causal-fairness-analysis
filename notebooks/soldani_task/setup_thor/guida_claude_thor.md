# Workflow Thor — FairMind benchmark con vLLM locale

## 0. Setup iniziale (una tantum)

```bash
ssh name.surname@thor.supsi.ch

mkdir -p /mnt/beegfs/scratch/name.surname/project1
cd /mnt/beegfs/scratch/name.surname/project1

# Copia qui i file: vllm_server.def, vllm_server.sbatch, jupyter_env.def
# (es. via scp dal tuo Mac, oppure git clone del repo tesi)

# Build dei due container (richiede accesso a Docker Hub dal login node — verifica coi sysadmin se non funziona)
apptainer build vllm_server.sif vllm_server.def
apptainer build jupyter_env.sif jupyter_env.def
```

## 1. Avvia il server vLLM (job persistente)

```bash
cd /mnt/beegfs/scratch/name.surname/project1
sbatch vllm_server.sbatch
squeue -u $USER          # prendi nota del JOBID
```

Aspetta che lo stato passi a `R`, poi controlla su quale nodo gira e che il
server sia pronto:

```bash
cat vllm_server_<JOBID>.out      # mostra "Job in esecuzione su: gnodeXX"
tail -f vllm_server_<JOBID>.err  # aspetta la riga "Uvicorn running on http://0.0.0.0:8000"
```

Annota l'hostname (es. `gnode05`) — ti serve nel notebook.

## 2. Sessione interattiva per Jupyter (job separato)

```bash
srun --partition=compute --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
```

Una volta dentro al nodo assegnato (es. `cnode02`):

```bash
cd /mnt/beegfs/scratch/name.surname/project1
apptainer exec jupyter_env.sif jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Annota anche qui l'hostname del nodo compute (es. `cnode02`) e il token
stampato da Jupyter.

## 3. Tunnel SSH dal tuo Mac

In un terminale **locale** (non su Thor):

```bash
ssh -L 8888:cnode02:8888 name.surname@thor.supsi.ch
```

Ora apri `http://localhost:8888` nel browser, incolla il token: sei dentro
Jupyter, ma il kernel gira su Thor.

> I due nodi (vLLM e compute) sono sulla stessa rete interna del cluster,
> quindi dal notebook puoi chiamare `http://gnode05:8000/v1` direttamente
> senza bisogno di un secondo tunnel.

## 4. Modifica il notebook

Applica le modifiche in `notebook_patch.py`:
- sostituisci la cella `client = OpenAI()` con l'inizializzazione che punta
  a `http://<hostname-vllm>:8000/v1`
- sostituisci la funzione `call_gpt` con la versione che usa
  `chat.completions.create` invece di `responses.create`

## 5. Fine sessione

```bash
# Quando hai finito di lavorare al notebook:
exit                         # esce dalla sessione srun interattiva (jupyter)

# Il server vLLM resta attivo per il job successivo; quando non ti serve più:
scancel <JOBID_vllm>
```

## Note importanti

- Il modello scaricato da HuggingFace finisce in `/scratch/hf_cache`
  (bind nello sbatch) — la prima `vllm serve` impiega qualche minuto per
  scaricare i pesi, le successive sono rapide (cache).
- `/scratch` non è backuppato: se vuoi tenere i risultati del benchmark
  (`benchmark_results/*.json`), copiali su `/home` o fuori dal cluster.
- Se il nodo compute e il nodo GPU non si vedono a vicenda per qualche
  motivo di rete/firewall interno, fallo presente — in quel caso l'opzione
  più semplice è tornare a lanciare tutto nello stesso job (server vLLM in
  background + jupyter nello stesso allocation).
