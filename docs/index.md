---
icon: lucide/rocket
---

# Causal AI Fairness

This project operationalizes causal fairness concepts in AI through a symbolic DSL, Bayesian network estimation, and effect decomposition tooling for both synthetic and real-world datasets.

## What is in this repository

- A symbolic language for counterfactual variables, events, and fairness queries.
- Bayesian network fitting utilities built on top of `pgmpy`.
- Effect estimators for total, spurious, direct, and indirect effects.
- Visualization helpers for fairness graphs and Sankey decompositions.
- Processed benchmark datasets and notebooks for experimentation.

## Install and run

```bash
uv sync
```

To preview the documentation locally:

```bash
uv run zensical serve
```

To build the static site:

```bash
uv run zensical build
```

## Quick example

```python
import pandas as pd

from src.effects import total_effect
from src.graph import build_sfm
from src.model import fit_discrete_bayesian_model

sfm = build_sfm(
  sensitive_attr="A",
  outcome_attr="Y",
  confounder_attrs=["Z"],
  mediator_attrs=["M"],
)

data = pd.read_csv("data/processed/adult.csv")
model = fit_discrete_bayesian_model(sfm, data, estimator_instance=(None, {}))
effect = total_effect(model, target=("Y", 1), private_attr="A", x0=0, x1=1)
print(effect)
```

## Documentation map

- Start with the API overview in [api/index.md](api/index.md).
- Use [api/model.md](api/model.md) for model fitting and network utilities.
- Use [api/effects.md](api/effects.md) for causal effect estimators.
- Use [api/sym_dsl.md](api/sym_dsl.md) for the symbolic DSL and counterfactual terms.
- Use [api/utils.md](api/utils.md) for graph and visualization helpers.

## Notes

The API reference is generated with `mkdocstrings`, so improving Python docstrings in `src/` will directly improve the rendered documentation.
