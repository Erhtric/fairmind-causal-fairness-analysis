# Thor cluster setup

Container definitions, Slurm submission scripts and documentation for running
the `soldani_task` notebooks on Thor, the SUPSI HPC cluster.

## Where to start

| Document | Purpose |
|---|---|
| [`thor-running-notebooks.md`](thor-running-notebooks.md) | **Operational guide for this project.** Directory layout, how to start the inference server, how to run a notebook in batch or interactively, troubleshooting. |
| [`thor-getting-started.md`](thor-getting-started.md) | General introduction to Thor: Slurm concepts, partitions, Apptainer, hardware reference. Not specific to this project. |

New to the cluster: read `thor-getting-started.md` first. Returning to run an
experiment: go straight to `thor-running-notebooks.md`.

## Files in this directory

### Container definitions

| File | Builds | Purpose |
|---|---|---|
| `llama_server.def` | `llama_server_gpu.sif` | llama.cpp with CUDA, for the inference server |
| `jupyter_env.def` | `jupyter_env.sif` | Python environment for notebook execution |

Both images are built once and live on scratch, under
`/mnt/beegfs/scratch/marco.soldani/causal_fairness/containers/`.

### Submission scripts

| Script | Runs | Requests a GPU |
|---|---|---|
| `llama_server.sbatch` | Persistent llama.cpp server, port 8080 | yes, one |
| `run_benchmark.sbatch` | `2_3_benchmark_thor.ipynb` | no |
| `run_benchmark_multi.sbatch` | `2_4_benchmark_multi_dataset.ipynb` | no |
| `run_report_pipeline.sbatch` | `2_6_report_pipeline.ipynb` | no |
| `run_complete_pipeline.sbatch` | `2_7_complete_report_pipeline.ipynb` | no |
| `run_extensive_benchmark.sbatch` | `2_8_extensive_benchmark.ipynb` | no |

The five notebook jobs do **not** start a server of their own. They locate the
already-running `llama_server_gpu` job with `squeue`, read its node, and connect
to it over HTTP. Start the server first, or they exit with an explanatory error.

### Deprecated

`notebook_patch.py` is a historical snippet: it targets a notebook name that no
longer exists (`2_1_benchmark_thor.ipynb`) and a 7B model, whereas the
experiments use Qwen2.5-14B. Kept only for reference; do not apply it.

## Model

| | |
|---|---|
| Model | Qwen2.5-14B-Instruct, Q4\_K\_M quantisation |
| Weights | `/mnt/beegfs/scratch/marco.soldani/causal_fairness/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| Server | llama.cpp, context size 131072, port 8080 |
| Slurm job name | `llama_server_gpu` |

Scratch is not backed up. The weights and both container images live there and
can be rebuilt or re-downloaded; **experiment results must be copied back to
`/home` or off the cluster**, or they are lost when scratch is cleaned.

## Prerequisites

Access to `thor.supsi.ch` requires the SUPSI network. From outside, connect to
the institutional VPN first: the hostname does not resolve otherwise.
