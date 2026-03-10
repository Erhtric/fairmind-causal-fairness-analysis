# causal-ai-fairness
This repo contains an operationalization of causal fairness concepts in AI. Synthetic and real-world experiments are included.

## Repository structure (in progress)
- `src/causalfairness`: new active implementation of the Streamlit app and effect computation modules.
- `src/causality`: new active implementation of causal graph utilities.
- `data/datasets`: local dataset artifacts used for experiments.
- `legacy/causalfairness` and `legacy/causality`: preserved snapshot of pre-migration code.

## Setup and run
- Install dependencies (recommended): `uv sync`
- Fallback install: `python -m pip install -e .`
- Run new app path: `streamlit run src/causalfairness/ui.py --server.enableCORS false --server.enableXsrfProtection false`
- Legacy app path (temporary): `streamlit run causalfairness/ui.py --server.enableCORS false --server.enableXsrfProtection false`

## Utilization pipeline
- See `docs/pipeline.md` for the full flow from dataset ingestion to causal decomposition and LLM narrative generation.

## Build a pgmpy Bayesian Network from data
- New module: `src/causalfairness/pgmpy_bn.py`
- Main API: `build_fitted_bn(df, edges, estimator="mle", estimator_kwargs=None, nodes=None, dropna=False)`
- Supported estimators: `"mle"` and `"bayesian"`
- Graph input: explicit edge list `[(parent, child), ...]` (or `networkx.DiGraph`)
- Alternative graph helper: `edges_from_parent_map({"child": ["parent1", "parent2"]})`
- Data requirement: all graph-node columns must be discrete (`bool`, `int`, `category`, `string`/`object`)
- Missing values: raise clear error by default, or set `dropna=True`

Example:
```python
from causalfairness.pgmpy_bn import build_fitted_bn

edges = [("X", "W"), ("X", "Y"), ("W", "Y")]
model = build_fitted_bn(df, edges, estimator="mle")
cpds = model.get_cpds()
```

# 1. Basic survey
Everything has to be contextualized based on SFM:
1. Deline an exhaustive list of effects coming from causal literature, in particular the ones coming from causal fairness works. (partially)
2. In a similar manner, deline the normative criterions that are used to define "fairness" in causal terms (demographic parity, eq of odds, etc.)

# 2. Syntethic experiment
1. Given binary variables and the Markovian SFM, produce a dataset of observations.
2. Compute all effects that are coming from literature (identifiability equations)
   
# 3. Real-world experiment
For each dataset in the ones used in AEQUITAS (resources can be found https://aiod.eu):
1. Intepretazione dell'analisi osservazione in termini causali
2. Calcolare effetti causali

# 4. Experiments
1. Extension to multi-varied discrete variable (>2 X, >2 Y). The idea is to take inspiration from works that discussed extension to PN, PS and PNS.
2. Extension to multi-varied discrete (ordered) variable
3. Extension to multi-varied continuos variables
4. Extension to NON-linear models
