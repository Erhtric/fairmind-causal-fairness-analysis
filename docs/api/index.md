# API Reference

The API reference covers the main public modules in the repository and is generated with `mkdocstrings` from the source code under `src/`.

## Core modules

- [Model API](model.md) documents Bayesian network fitting.
- [Effects API](effects.md) documents numeric and symbolic fairness effect estimators.
- [Symbolic DSL API](sym_dsl.md) documents variables, events, counterfactual terms, and queries.
- [Utilities API](utils.md) documents graph construction and visualization helpers.

## Package layout

```text
src/
├── effects.py
├── graph.py
├── model.py
├── preprocess.py
├── sym/
│   ├── dsl.py
│   ├── ctf_calculus.py
│   └── effects.py
└── visualisation/
	├── graph.py
	└── sankey.py
```

## Usage pattern

Most workflows in this project follow the same path:

1. Build or inspect a structural fairness model graph.
2. Fit a discrete Bayesian network from observed data.
3. Compute fairness-relevant effects numerically or symbolically.
4. Visualize the resulting decomposition if needed.
