# Project Guidelines

## Code Style
- Primary language is Python; active code lives under `src/causalfairness/` and `src/causality/`.
- Legacy snapshot is preserved under `legacy/causalfairness/` and `legacy/causality/`; avoid editing legacy unless explicitly requested.
- Follow existing import style in touched files (many modules use `from ... import *`, e.g. `src/causalfairness/functions.py`). Do not refactor import style unless required by the task.
- Preserve effect-domain naming already used across the project (`te`, `de`, `ie`, `se`, `x0`, `x1`, `w_cols`, `z_cols`).
- Prefer small helper functions and DataFrame-first transformations, matching `src/causalfairness/functions.py` and `src/causalfairness/ui.py`.
- Type hints are partial; add them only where nearby code already uses them (example: `src/causality/graph.py`).

## Architecture
- `src/causalfairness/ui.py` is the Streamlit entrypoint and orchestration layer (data upload, variable-role selection, analysis execution, visualization, LLM narrative).
- `src/causalfairness/functions.py` contains core effect computation and decomposition logic (frequentist + Bayesian helpers).
- `src/causalfairness/general_effects.py` defines effect equations used by the computation layer.
- `src/causalfairness/sankey.py` and Graphviz usage in `ui.py` handle visualization outputs.
- `src/causality/graph.py` is a separate causal-graph utility module; keep it decoupled from Streamlit UI concerns.

## Build and Test
- Python requirement from `pyproject.toml`: `>=3.13,<3.14`.
- Install dependencies (recommended): `uv sync`.
- Fallback install if `uv` is unavailable: `python -m pip install -e .`.
- Run app: `streamlit run src/causalfairness/ui.py --server.enableCORS false --server.enableXsrfProtection false`.
- Tests are not currently defined (no test suite/config discovered); do not invent new global test commands.

## Project Conventions
- Keep the SFM/fairness terminology and decomposition semantics used in `README.md` and `src/causalfairness/ui.py`.
- Respect UI role exclusivity logic (X/Y/W/Z assignment and session-state callbacks) implemented in `src/causalfairness/ui.py`.
- Preserve support for unknown mediator/confounder order handling (`unknown_w_order`, `unknown_z_order`) and interval outputs in `src/causalfairness/functions.py`.
- Reuse shared library imports from `src/causalfairness/libraries.py` when adding dependencies used across multiple modules.
- This repo is notebook-heavy (`legacy/causalfairness/*.ipynb`); when changing shared logic, ensure notebook flows remain compatible.

## Integration Points
- OpenAI integration is in `src/causalfairness/ui.py` via `OpenAI` client and prompt templates from `src/causalfairness/resources/fairmind_prompts.txt`.
- Streamlit is the main runtime/UI framework; prefer Streamlit-native state and caching patterns (`st.session_state`, `@st.cache_data`).
- Data inputs are user-uploaded files (`csv`, `tsv`, `xlsx`, `json`) parsed in the UI layer; reference datasets are in `data/datasets/`.

## Security
- `OPENAI_API_KEY` is required and loaded from `.env`; do not hardcode API keys.
- Treat uploaded dataset content as sensitive; avoid logging raw rows unless strictly necessary.
- The devcontainer run command disables CORS/XSRF for local development; do not assume those flags are safe for production deployment.
