# Causal AI Fairness

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An operationalisation of causal fairness concepts presented in [Causal Fairness Analysis](https://causalai.net/r90.pdf) by Drago Plečko and Elias Bareinboim. The library further implements a "interpretative" pipeline, following the ideas from "LLM as Data Scientist", to use LLM to interpret the causal fairness metrics computed.

## 🌟 Key Features

- **Causal Effect Estimation**: Implementation of causal fairness metrics belonging to the total variation family under observational identifiable conditions:
  - **Total Variation (TV)** and **Total Effect (TE)**
  - **Natural Direct Effect (NDE)** and **Natural Indirect Effect (NIE)**
  - **Spurious Effect (SE)**
- **Effect Decomposition**: Advanced methods to decompose indirect and spurious effects into variable-specific contributions.
- **Bayesian Network Integration**: seamless fitting of discrete Bayesian Networks to data using `pgmpy`.
- **Automated Reporting**: Tools to generate comprehensive causal fairness reports in tidy formats and LaTeX.
- **Visualizations**: Sankey diagrams and causal graph plots.

## 📂 Repository Structure

```text
├── src/                # Core library source code
│   ├── visualisation/ # Graph and Sankey visualization utilities
│   ├── model.py       # Bayesian Network fitting logic
│   ├── effects.py     # Causal effect estimation and decomposition
│   └── llm.py         # LLM integration for automated analysis
├── notebooks/          # Exploratory analysis and usage examples
├── data/               # Local datasets (Adult, COMPAS, German Credit, etc.)
├── experiments/        # Research experiments and LaTeX reports
└── ui/                 # Streamlit application for interactive analysis
```

## 🚀 Getting Started

### Installation

The project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/Erhtric/causal-ai-fairness.git
cd causal-ai-fairness

# Sync dependencies and create virtual environment
uv sync
```

*Fallback:* `pip install -e .`

### Quick Start: Causal Effect Estimation

The core workflow is: define a structural fairness model, fit a discrete Bayesian network, then compute the effect metrics you care about.

```python
import pandas as pd
from pgmpy.estimators import DiscreteBayesianEstimator

from src.effects import DE, IE, SE, TE, TV, compute_fairness_report
from src.graph import build_sfm
from src.model import fit_discrete_bayesian_model

# 1. Prepare a discrete dataset.
# Replace this with one of the processed datasets in data/ or your own categorical data.
df = pd.DataFrame(
  {
    "X": ["x0", "x0", "x1", "x1", "x0", "x1"],
    "W": ["w0", "w1", "w0", "w1", "w0", "w1"],
    "Y": ["y0", "y0", "y1", "y1", "y0", "y1"],
  }
).astype("category")

# 2. Build a structural fairness model.
sfm = build_sfm(
  sensitive_attr="X",
  outcome_attr="Y",
  confounder_attrs=[],
  mediator_attrs=["W"],
)

# 3. Fit a discrete Bayesian network to the data.
bn = fit_discrete_bayesian_model(
  sfm=sfm,
  data=df,
  estimator_instance=(
    DiscreteBayesianEstimator,
    {"prior_type": "dirichlet", "pseudo_counts": 1.0},
  ),
)

assert bn.check_model()

# 4. Compute fairness effects for a target outcome state.
target = ("Y", "y1")
x0 = "x0"
x1 = "x1"

results = pd.Series(
  {
    "TV": TV(bn, target, "X", x0, x1),
    "TE": TE(bn, target, "X", x0, x1),
    "SE(x0)": SE(bn, target, "X", x0),
    "SE(x1)": SE(bn, target, "X", x1),
    "NDE": DE(bn, target, "X", x0, x1),
    "NIE": IE(bn, target, "X", x1, x0),
  }
)

print(results.round(6))

# 5. Build a tidy report table if you want a single summary object.
report = compute_fairness_report(bn, target, "X", x0, x1)
print(report)
```

For a real analysis, replace the toy `DataFrame` with one of the processed datasets under `data/`, then adapt the node names and states to match the variables in your graph. All variables passed to the Bayesian network must be discrete.

## 📖 References

This implementation is primarily based on the theoretical framework established in:
1. D. Plečko and E. Bareinboim, “Causal Fairness Analysis: A Causal Toolkit for Fair Machine Learning,” FNT in Machine Learning, vol. 17, no. 3, pp. 304–589, 2024, doi: 10.1561/2200000106.



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
