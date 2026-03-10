# causal-ai-fairness

This repo contains an operationalization of causal fairness concepts in AI. Synthetic and real-world experiments are included.

## Repository structure

- `src/`: Core library implementation.
  - `model.py`: Bayesian Network fitting using `pgmpy`.
  - `effects.py`: Causal effect estimation logic.
  - `sym/`: Symbolic Causal DSL and calculus.
  - `visualisation/`: Visualization utilities (Sankey diagrams).
- `data/datasets`: Local dataset artifacts used for experiments.
- `legacy/causalfairness`: Preserved snapshot of pre-migration code (contains the current Streamlit UI).

## Setup and run

1. **Install dependencies** (recommended):
   ```bash
   uv sync
   ```
   Fallback: `python -m pip install -e .`

2. **Run the application**:
   Currently, the UI is located in the legacy directory:
   ```bash
   streamlit run legacy/causalfairness/ui.py --server.enableCORS false --server.enableXsrfProtection false
   ```

## Documentation

The project uses MkDocs with Material theme for documentation, including full API reference generated from code docstrings.

**View documentation locally**:
```bash
uv run mkdocs serve
```
Then open http://127.0.0.1:8000/ in your browser.

**Build static documentation**:
```bash
uv run mkdocs build
```
This generates the static site in the `site/` directory.

The documentation includes:
- User guide and quick start
- Complete API reference for all modules
- Code examples and usage patterns

## Build a pgmpy Bayesian Network from data

- Module: `src/model.py`
- Main API: `fit_discrete_bayesian_model(sfm, data, estimator_instance)`

Example:

```python
import networkx as nx
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from src.model import fit_discrete_bayesian_model

# 1. Define the Structural Fairness Model (SFM)
sfm = nx.DiGraph([("X", "W"), ("X", "Y"), ("W", "Y")])

# 2. Prepare data
df = pd.read_csv("data.csv")

# 3. Fit the model
# estimator_instance is a tuple: (EstimatorClass, kwargs)
model = fit_discrete_bayesian_model(
    sfm=sfm,
    data=df,
    estimator_instance=(MaximumLikelihoodEstimator, {})
)
```

# 1. Basic survey
Everything has to be contextualized based on SFM:
1. Delineate an exhaustive list of effects coming from causal literature, in particular the ones coming from causal fairness works. (partially)
2. In a similar manner, delineate the normative criterions that are used to define "fairness" in causal terms (demographic parity, eq of odds, etc.)

# 2. Synthetic experiment
1. Given binary variables and the Markovian SFM, produce a dataset of observations.
2. Compute all effects that are coming from literature (identifiability equations)

# 3. Real-world experiment
For each dataset in the ones used in AEQUITAS (resources can be found https://aiod.eu):
1. Interpretation of observational analysis in causal terms.
2. Compute causal effects.

# 4. Experiments
1. Extension to multi-varied discrete variable (>2 X, >2 Y). The idea is to take inspiration from works that discussed extension to PN, PS and PNS.
2. Extension to multi-varied discrete (ordered) variable.
3. Extension to multi-varied continuous variables.
4. Extension to NON-linear models.
