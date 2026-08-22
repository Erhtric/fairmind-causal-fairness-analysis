# Running notebooks on the Thor cluster

Operational guide for executing the `soldani_task` notebooks on Thor, the SUPSI
HPC cluster. It covers the layout of the working directories, the two execution
modes (batch and interactive), and the cluster-specific constraints that shaped
the submission scripts.

Paths and job names below were verified directly on the cluster rather than
assumed. Where a fact may have drifted since, the guide gives the command to
re-check it instead of restating it.

---

## 1. Directory layout

Two locations, with different guarantees. Keeping the distinction is what
prevents work from being lost.

### `/home/marco.soldani/causal_fairness/`

The git clone: source (`src/`), notebooks (`notebooks/soldani_task/`) and
submission scripts (`notebooks/soldani_task/setup_thor/`). Small files, backed
up, version-controlled. **Anything that must survive belongs here.**

Confirm which branch is checked out before relying on it:

```bash
git -C /home/marco.soldani/causal_fairness branch --show-current
```

### `/mnt/beegfs/scratch/marco.soldani/causal_fairness/`

Large, transient, **not backed up**:

| Path | Contents |
|---|---|
| `containers/llama_server_gpu.sif` | GPU container (llama.cpp + CUDA) |
| `containers/jupyter_env.sif` | Notebook execution container |
| `models/Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Model weights |
| `slurm/llama_server_gpu.sbatch` | Copy of the server script |

> **Scratch is not a source of truth.** Any script edited directly on scratch
> must be copied back into the repository, or it is lost the next time scratch
> is cleaned. This has happened once: `slurm/llama_server_gpu.sbatch` ran the
> live server for days while being tracked nowhere, and was recovered into the
> repository as `setup_thor/llama_server.sbatch` on 2026-07-20.

---

## 2. Components

| Item | Value |
|---|---|
| Model | Qwen2.5-14B-Instruct, Q4\_K\_M quantisation |
| Inference server | llama.cpp, context size 131072, port 8080 |
| Server job name | `llama_server_gpu` |
| Login host | `marco.soldani@thor.supsi.ch` |

The server is a long-lived job (`--time=14-00:00:00`) holding one GPU. Notebook
jobs do **not** request a GPU of their own: they locate the running server by
job name and connect to it over HTTP.

> `thor.supsi.ch` resolves only from the SUPSI network. From outside, connect to
> the institutional VPN first, or the hostname will not resolve.

---

## 3. Submission scripts

All four live in `setup_thor/` and follow the same pattern: find the running
server, export `LLAMA_HOST` / `LLAMA_PORT`, execute a notebook through
`nbconvert` inside `jupyter_env.sif`.

| Script | Notebook | Walltime | On cell error |
|---|---|---|---|
| `run_benchmark.sbatch` | `2_3_benchmark_thor.ipynb` | 2 h | job fails |
| `run_benchmark_multi.sbatch` | `2_4_benchmark_multi_dataset.ipynb` | 4 h | `--allow-errors`: job continues |
| `run_report_pipeline.sbatch` | `2_6_report_pipeline.ipynb` | 1 h | job fails |
| `run_complete_pipeline.sbatch` | `2_7_complete_report_pipeline.ipynb` | 2 h | job fails |

`run_complete_pipeline.sbatch` asks for two hours rather than one because it makes
three requests to the model instead of one, and the middle one enumerates twenty five
pairs of confounder and mediator states step by step: on Adult that took 131 seconds
and about 9700 output tokens, against eleven seconds for each report.

The difference in the last column is deliberate. A benchmark sweep over ten
datasets should not abort because one dataset fails, whereas a single reporting
run that fails silently would produce an output notebook with a red cell that
nobody notices. `run_report_pipeline.sbatch` therefore omits `--allow-errors`
and checks the exit status.

---

## 4. First-time setup: build the notebook container

Required only once, or after `jupyter_env.def` changes.

```bash
ssh marco.soldani@thor.supsi.ch
mkdir -p /mnt/beegfs/scratch/marco.soldani/causal_fairness/containers
apptainer build \
    /mnt/beegfs/scratch/marco.soldani/causal_fairness/containers/jupyter_env.sif \
    /home/marco.soldani/causal_fairness/notebooks/soldani_task/setup_thor/jupyter_env.def
```

The build takes a few minutes and is fine on the login node (`hnode01`), since
`jupyter_env.def` compiles nothing. `llama_server.def` does compile llama.cpp
and is better built on a compute node.

Note that `pgmpy` pulls in numpy, scipy, scikit-learn, torch and pyro-ppl
transitively, so the build downloads considerably more than the explicit `pip`
list suggests.

---

## 5. Step 1: check whether the inference server is running

```bash
ssh marco.soldani@thor.supsi.ch
squeue -u $USER --name=llama_server_gpu
```

- State `R` (RUNNING): the server is up; skip to step 3.
- Nothing listed, or state `PD` / `CD` / `F`: continue to step 2.

## 6. Step 2: start the inference server

Only if step 1 found nothing running.

```bash
cd /home/marco.soldani/causal_fairness/notebooks/soldani_task/setup_thor
sbatch llama_server.sbatch
squeue -u $USER
```

Note the JOBID. Once the job reaches state `R`, confirm the server is
answering. Log files are written to the directory `sbatch` was invoked from:

```bash
cat llama_server_gpu_<JOBID>.out     # reports the assigned node
tail -f llama_server_gpu_<JOBID>.err # wait for the server to accept requests
```

## 7. Step 3: execute a notebook

### Option A: batch submission (preferred for recorded runs)

```bash
cd /home/marco.soldani/causal_fairness/notebooks/soldani_task/setup_thor
sbatch run_report_pipeline.sbatch
squeue -u $USER
```

The job should reach state `R` almost immediately: it runs on the `gpu`
partition but without `--gres=gpu:1`, so it competes for CPU and memory only.

The executed notebook is written to `notebooks/soldani_task/` as
`report_pipeline_output_<JOBID>.ipynb` (or `benchmark_output_<JOBID>.ipynb`),
with all cell outputs retained.

Two errors indicate a skipped step:

| Message | Cause |
|---|---|
| `nessun job 'llama_server_gpu' in stato RUNNING trovato` | Server not started. Return to step 1 |
| `containers/jupyter_env.sif non esiste ancora` | Container not built. Return to section 4 |

### Option B: interactive Jupyter (preferred for debugging)

On Thor:

```bash
srun --partition=compute --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
cd /home/marco.soldani/causal_fairness/notebooks/soldani_task
apptainer exec \
    /mnt/beegfs/scratch/marco.soldani/causal_fairness/containers/jupyter_env.sif \
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Note the assigned node (for example `cnode02`) and the printed token. Then, from
a local terminal, forward the port:

```bash
ssh -L 8888:cnode02:8888 marco.soldani@thor.supsi.ch
```

Open `http://localhost:8888` and supply the token.

Before running any cell, point the notebook at the inference server. The node
name is the `NODELIST` column of the `llama_server_gpu` job:

```python
import os
os.environ["LLAMA_HOST"] = "gnode04"
os.environ["LLAMA_PORT"] = "8080"
```

The notebooks read these variables automatically and fall back to
`localhost:8080`, so the same code runs unchanged locally and on the cluster.

## 8. Step 4: monitoring and cleanup

```bash
squeue -u $USER                                        # all jobs
scontrol show job <jobid>                              # why a job is PENDING
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # accounting
scancel <jobid>                                        # cancel
```

The inference server may be left running across days if it is still needed, but
it holds one GPU on the `gpu` partition for as long as it lives. Cancel it with
`scancel` once finished, so the GPU returns to other users. Interactive jobs
started with `srun --pty` terminate on `exit`.

---

## 9. Cluster constraints behind the script design

These are properties of the cluster, not of this project. They explain choices
in the submission scripts that would otherwise look arbitrary, and they are the
first thing to check when a job fails in an unexplained way.

### User resolution is broken on the `compute` partition

Verified 2026-07-20 with a dedicated debug job. Inside a batch job on
`compute`, NSS/LDAP user lookup fails: even `whoami` returns
`cannot find name for user ID 30117812`. On the `gpu` partition it works
correctly. This is cluster infrastructure, not a container or script problem.

Three consequences, all already applied:

1. Notebook jobs run on `partition=gpu` **without** `--gres=gpu:1`. This avoids
   the NSS failure without consuming a GPU.
2. The server is located with `squeue --name=llama_server_gpu`, never with
   `-u $USER`. The user filter can fail with `Invalid user: marco.soldani` for a
   perfectly valid user, producing a false "no server found".
3. `apptainer exec` is invoked with `--no-home`. Without it, apptainer fails
   with `Couldn't determine user account information` before the container even
   starts, because it tries to resolve the host user to bind the home directory.
   The home directory is bound explicitly instead.

If `unknown userid` or `Couldn't determine user account information` appears on
**any** partition in future, it is most likely this same infrastructure issue
and worth reporting to the Thor administrators rather than working around in
the scripts.

### Exit codes must be checked explicitly

`apptainer exec` failures were once not checked, so a job could finish with
`ExitCode 0:0` and the message "Benchmark completato" while the notebook had
never run and no output file existed. All three scripts now test the exit
status and fail loudly.

### LaTeX is not available in the container

`pdflatex` is not installed in `jupyter_env.sif`. The verification cell in
`2_6_report_pipeline.ipynb` detects this with `shutil.which()` and skips the
check rather than failing, leaving `latex_compiles` as `null` in the result
JSON. To verify compilation, download the `.tex` and build it locally:

```bash
pdflatex -interaction=nonstopmode <file>.tex
```

---

## 10. Troubleshooting

**A job stays PENDING.**
Inspect `scontrol show job <jobid>` and read the `Reason` field. Notebook jobs
request no GPU, so GPU contention should not be the cause.

**`Connection error` when the notebook calls the model.**
Either the `llama_server_gpu` job is not running, or `LLAMA_HOST` points at the
wrong node. Repeat step 1 and re-read the `NODELIST` column.

**`apptainer build` fails on the first or last line of a `.def` file.**
Check that the definition file does not contain stray `cat > ... << EOF` or
`EOF` lines left over from a terminal copy-paste. This was a real defect in
`llama_server.def`, since corrected.

**A stray copy of a notebook under `scratch/`.**
The authoritative copy is the one in the git repository, under
`/home/marco.soldani/causal_fairness/notebooks/soldani_task/`; that is the one
the submission scripts execute. Copies under scratch are residue from manual
testing and can be deleted.