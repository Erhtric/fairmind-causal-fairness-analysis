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

```python
import networkx as nx
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from src.model import fit_discrete_bayesian_model
from src.effects import compute_fairness_report

# 1. Define the Structural Fairness Model (SFM)
# Annotate nodes with types: sensitive, mediator, confounder, target
sfm = nx.DiGraph()
sfm.add_node("Sex", type="sensitive")
sfm.add_node("Major", type="mediator")
sfm.add_node("Admission", type="target")
sfm.add_edges_from([("Sex", "Major"), ("Sex", "Admission"), ("Major", "Admission")])

# 2. Fit the model to data
df = pd.read_csv("data/datasets/berkeley_filtered.csv")
model = fit_discrete_bayesian_model(
    sfm=sfm, 
    data=df, 
    estimator_instance=(MaximumLikelihoodEstimator, {})
)

# 3. Compute fairness report
report = compute_fairness_report(
    bn=model,
    target=("Admission", "Accepted"),
    private_attr="Sex",
    x0="Female",
    x1="Male"
)
print(report)
```

## 📖 References

This implementation is primarily based on the theoretical framework established in:
1. D. Plečko and E. Bareinboim, “Causal Fairness Analysis: A Causal Toolkit for Fair Machine Learning,” FNT in Machine Learning, vol. 17, no. 3, pp. 304–589, 2024, doi: 10.1561/2200000106.



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
