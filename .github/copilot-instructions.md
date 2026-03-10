# Copilot Instructions

## Build, Test, and Lint

- **Dependency Management**: This project uses `uv`.
  - Install dependencies: `uv sync`
  - Add dependency: `uv add <package>`
- **Testing**: Run tests with `pytest`.
  - Run all tests: `uv run pytest`
  - Run a specific test file: `uv run pytest tests/test_model.py`
- **Linting & Formatting**: This project uses `ruff`.
  - Check: `uv run ruff check .`
  - Format: `uv run ruff format .`

## High-Level Architecture

The project implements causal fairness concepts using a symbolic DSL and Bayesian Networks.

- **`src/sym/`**: Contains the symbolic Domain Specific Language (DSL) for causal inference.
  - `dsl.py`: Defines core primitives (`Variable`, `CounterfactualTerm`, `Event`, `Query`).
  - Supports syntax sugar like `Y @ {X: 1}` for counterfactuals ($Y_{X=1}$).
- **`src/model.py`**: Handles fitting Discrete Bayesian Networks (using `pgmpy`) to data based on a structural model.
- **`src/visualisation/`**: Visualization utilities, primarily Sankey diagrams (`sankey.py`) using `plotly` to show effect decomposition.
- **`src/effects.py`**: Logic for computing causal effects via adjustment and inference on Bayesian Networks.
- **`legacy/`**: Contains preserved pre-migration code. Avoid modifying unless necessary.

## Key Conventions

- **Symbolic DSL**: When working with causal queries, use the DSL defined in `src/sym/dsl.py`.
  - Variables are immutable and hashable.
  - Counterfactuals are created with the `@` operator: `Var @ {Intervention: Value}`.
  - Events are conjunctions of atomic propositions.
- **Data Handling**:
  - `pandas` DataFrames are used for datasets.
  - `networkx.DiGraph` is used for structural models.
- **Typing**: Type hints are used partially. Add them for new code.
- **Logging**: Use `loguru` for logging.
- **Visualization**: Use `plotly.graph_objects` for interactive plots (Sankey diagrams).

## Project Structure

```
src/
├── model.py          # Bayesian Network fitting
├── effects.py        # Effect calculation
├── graph.py          # Graph utilities
├── preprocess.py     # Data preprocessing
├── sym/              # Symbolic Causal DSL
│   ├── dsl.py
│   ├── ctf_calculus.py
│   └── effects.py
└── visualisation/    # Plotting
    ├── sankey.py
    └── graph.py
```
