"""Shared helpers for the benchmark notebooks.

These functions used to live inside the notebooks: run_fairmind was repeated in
five of them and had drifted apart, while build_llm_prompt and
compute_discrepancies existed only in 2_3_benchmark_thor. Both misalignments
reported in the experimental chapter started there, fixed in one copy and left
standing in the others.

The code was moved without changing its behaviour. tests/test_benchmark_common.py
pins the reference values on Adult.
"""

from __future__ import annotations

import time
from itertools import product

import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .effects import (
    natural_direct_effect,
    natural_indirect_effect,
    total_effect,
    total_variation,
)
from .graph import build_sfm
from .model import fit_discrete_bayesian_model

###############################################################################
# Reference effects
###############################################################################

def discretise(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the discretisation declared in the configuration, in place.

    ``config["binning"]`` maps a column name to either a numeric specification
    with ``bins`` and ``labels``, applied through ``pd.cut``, or a
    ``mapping`` dictionary that rewrites categorical levels into coarser ones.
    Columns absent from the frame are skipped, so a configuration may describe
    more than a given extract contains.
    """
    for column, spec in config.get("binning", {}).items():
        if column not in df.columns:
            continue
        if "mapping" in spec:
            df[column] = df[column].map(spec["mapping"])
        else:
            df[column] = pd.cut(
                df[column],
                bins=spec["bins"],
                labels=spec["labels"],
                include_lowest=True,
            )
    return df


def run_fairmind(config: dict) -> tuple[dict, DiscreteBayesianNetwork, int, float]:
    df = pd.read_csv(config["csv_path"])
    cols = (
        [config["protected"]]
        + config["mediators"]
        + config["confounders"]
        + [config["target_col"]]
    )
    df = df[cols].dropna()

    # Binned here, in-place, once: the BN fitted on this binned data is the
    # SAME instance build_llm_prompt() queries to build the tables given to
    # the LLM, so both sides start from identical numbers on every cell.
    #
    # The binning comes from the configuration rather than being written out
    # here, so the same function serves every dataset. Adult declares exactly
    # what used to be hardcoded.
    #
    # Cardinality is what makes or breaks the prompt: sixteen education levels
    # against five hour bands is eighty (z,w) pairs, and at that size
    # Qwen2.5-14B never finished DE and IE. The answer was cut off even at
    # max_tokens=16384, and a compact output format made it worse, with
    # invented values such as TE identical to TV. Five education tiers bring
    # the pairs down to twenty five.
    df = discretise(df, config)

    sfm = build_sfm(
        sensitive_attr=config["protected"],
        outcome_attr=config["target_col"],
        confounder_attrs=config["confounders"],
        mediator_attrs=config["mediators"],
        sorted_mediators=len(config["mediators"]) > 1,
        sorted_confounders=len(config["confounders"]) > 1,
    )
    # Laplace smoothing with pseudo-count alpha = 1 on every state of every
    # variable, i.e. P(state | parents) = (count + 1) / (N_parents + n_states),
    # matching the parameter estimation described in the reference paper.
    # pgmpy's "K2" is a shorthand for exactly this; the explicit dirichlet form
    # is used here because it states alpha = 1 in the code.
    bn = fit_discrete_bayesian_model(
        sfm=sfm,
        data=df,
        estimator_instance=(
            BayesianEstimator,
            {"prior_type": "dirichlet", "pseudo_counts": 1},
        ),
    )

    target = (config["target_col"], config["target_val"])
    x0, x1 = config["x0"], config["x1"]

    start = time.perf_counter()
    tv = total_variation(bn, target, config["protected"], x0, x1)
    te = total_effect(bn, target, config["protected"], x0, x1)
    effects = {
        "TV": tv,
        "TE": te,
        # SE = TV - TE (Eq. 3, Plecko & Bareinboim 2024), the same identity
        # the prompt asks the LLM to apply.
        "SE": tv - te,
        "DE": natural_direct_effect(bn, target, config["protected"], x0, x1),
        # The paper defines the indirect effect twice, and the two are
        # different quantities, not a sign flip. Both are kept.
        #
        # "IE" is IE_{x0,x1} from Eq. 8, the one the computation prompt asks
        # the model for, so it is the one the discrepancy table compares.
        #
        # "IE_reverse" is IE_{x1,x0}, the form Eq. 9 puts in TE = DE - IE.
        # Only this one closes the identity, so the report and the recap
        # rules use it. Q2 flips if the direct form is substituted.
        "IE": natural_indirect_effect(bn, target, config["protected"], x0, x1),
        "IE_reverse": natural_indirect_effect(bn, target, config["protected"], x1, x0),
    }
    elapsed = time.perf_counter() - start

    return effects, bn, len(df), elapsed


###############################################################################
# Querying the network to build the prompt
###############################################################################

def _bn_states(bn, var: str) -> list:
    return bn.get_cpds(var).state_names[var]

def _bn_combos(bn, variables: list[str]) -> list[dict]:
    """All joint state combinations for a list of BN variables, as a list of
    dicts {variable: state}. Used to enumerate the table rows (one per
    combination) without hardcoding the states."""
    if not variables:
        return [{}]
    state_lists = [_bn_states(bn, v) for v in variables]
    # strict=True: product() yields tuples as long as state_lists, so a
    # length mismatch here would mean the states were built for a
    # different variable list.
    return [
        dict(zip(variables, combo, strict=True))
        for combo in product(*state_lists)
    ]

def build_llm_prompt(config: dict, bn, n_rows: int) -> str:
    """Builds the LLM prompt by querying DIRECTLY the Bayesian Network already
    fitted in run_fairmind() (same instance, same Laplace smoothing), instead of
    recomputing the probabilities with pandas on the raw dataset. Both sides
    then start from identical numbers on every table cell, including the
    sparsest ones (supervisor's Point 4).
    """
    protected = config["protected"]
    target_var = config["target_col"]
    target_val = config["target_val"]
    confounders = config["confounders"]
    mediators = config["mediators"]
    x0, x1 = config["x0"], config["x1"]

    ve = VariableElimination(bn)

    # --- 1. P(Y=y | X) ---
    rows = []
    for x in [x0, x1]:
        f = ve.query(variables=[target_var], evidence={protected: x}, show_progress=False)
        p = float(f.get_value(**{target_var: target_val}))
        rows.append({protected: x, "P(Y=y|X)": round(p, 4)})
    p_y_given_x = pd.DataFrame(rows)

    # --- 2. P(Z) - marginal distribution of the confounders ---
    z_factor = ve.query(variables=confounders, joint=True, show_progress=False)
    rows = []
    for z_combo in _bn_combos(bn, confounders):
        p = float(z_factor.get_value(**z_combo))
        rows.append({**z_combo, "P(Z)": round(p, 4)})
    p_z = pd.DataFrame(rows)

    # --- 3. P(Y=y | X, Z) ---
    rows = []
    for x in [x0, x1]:
        for z_combo in _bn_combos(bn, confounders):
            f = ve.query(variables=[target_var], evidence={protected: x, **z_combo}, show_progress=False)
            p = float(f.get_value(**{target_var: target_val}))
            rows.append({protected: x, **z_combo, "P(Y=y|X,Z)": round(p, 4)})
    p_y_given_xz = pd.DataFrame(rows)

    # --- 4. P(W | X, Z) ---
    rows = []
    for x in [x0, x1]:
        for z_combo in _bn_combos(bn, confounders):
            f = ve.query(variables=mediators, evidence={protected: x, **z_combo}, joint=True, show_progress=False)
            for w_combo in _bn_combos(bn, mediators):
                p = float(f.get_value(**w_combo))
                rows.append({protected: x, **z_combo, **w_combo, "P(W|X,Z)": round(p, 4)})
    p_w_given_xz = pd.DataFrame(rows)

    # --- 5. P(Y=y | X, W, Z) ---
    rows = []
    for x in [x0, x1]:
        for z_combo in _bn_combos(bn, confounders):
            for w_combo in _bn_combos(bn, mediators):
                f = ve.query(variables=[target_var], evidence={protected: x, **z_combo, **w_combo}, show_progress=False)
                p = float(f.get_value(**{target_var: target_val}))
                rows.append({protected: x, **z_combo, **w_combo, "P(Y=y|X,W,Z)": round(p, 4)})
    p_y_given_xwz = pd.DataFrame(rows)

    def to_compact_csv(d: pd.DataFrame) -> str:
        return d.to_csv(index=False)

    return f"""You are a causal fairness expert. Compute four causal fairness effects
using the Standard Fairness Model (SFM) by Plecko and Bareinboim (2024).

You are given PRE-AGGREGATED CONDITIONAL PROBABILITY TABLES computed from a fitted
Bayesian Network (n={n_rows} training rows, Laplace-smoothed CPDs, alpha=1). Use these tables
directly — do not assume access to raw data.
Note: "hours-per-week" has been discretized into bins: <=20, 21-35, 36-45, 46-60, >60.
Note: "education" has been grouped into tiers: <HS, HS-grad, Some-college, Bachelors, Grad.

VARIABLE ROLES:
- X (protected): "{protected}", x0="{x0}", x1="{x1}"
- Y (target):    "{target_var}", target state="{target_val}"
- W (mediators): {mediators}
- Z (confounders): {confounders}

TABLE 1 — P(Y=y | X):
{to_compact_csv(p_y_given_x)}

TABLE 2 — P(Z):
{to_compact_csv(p_z)}

TABLE 3 — P(Y=y | X, Z):
{to_compact_csv(p_y_given_xz)}

TABLE 4 — P(W | X, Z):
{to_compact_csv(p_w_given_xz)}

TABLE 5 — P(Y=y | X, W, Z):
{to_compact_csv(p_y_given_xwz)}

IDENTIFICATION FORMULAE (use these exactly, aggregating over TABLE rows as needed):
- TV = P(Y=y | X=x1) - P(Y=y | X=x0)                    [from TABLE 1]
- TE = sum_z [P(Y=y|x1,z) - P(Y=y|x0,z)] * P(z)          [from TABLE 3, TABLE 2]
- DE = sum_z,w [P(Y=y|x1,w,z) - P(Y=y|x0,w,z)] * P(w|x0,z) * P(z)   [from TABLE 5, TABLE 4, TABLE 2]
- IE = sum_z,w P(Y=y|x0,w,z) * [P(w|x1,z) - P(w|x0,z)] * P(z)       [from TABLE 5, TABLE 4, TABLE 2]

Note: the Spurious Effect (SE) is NOT requested here — it is fully determined
by SE = TV - TE (Plecko & Bareinboim, 2024), so it is derived afterwards
from your TV and TE values rather than computed independently.

INSTRUCTIONS:
TABLE 2 has 5 values of z and hours-per-week has 5 bins w, so DE and IE have
exactly 25 terms each. Compute every one of them.

Write one line per (z,w) term, showing the factors you read from the tables and
the resulting product. Look up TABLE 4 and TABLE 5 by z and w together.

Do NOT write "repeating similar calculations", "similarly for", "and so on", or any
other abbreviation, and do NOT report a subtotal for a z you have not written out
term by term first. A subtotal without its 5 terms above it is a wrong answer, even
if the number looks plausible.

Immediately before the final answer, state how many terms you wrote for DE and how
many for IE. Both counts must be 25. If either is below 25, compute the missing
terms before answering.

CHECKS (apply to every term before you sum):
1. Magnitude. Every factor you multiply is a probability between 0 and 1, so the
   product cannot be larger in absolute value than its smallest factor. Check each
   term against this bound. If |term| exceeds the smallest factor you have misplaced
   a decimal point: recompute that term before continuing.
2. Source table. TABLE 1 is used ONLY for TV. TE reads P(Y=y|x,z) from TABLE 3;
   DE and IE read P(Y=y|x,w,z) from TABLE 5 and P(w|x,z) from TABLE 4. Never take a
   value from TABLE 1 into a TE, DE or IE term: the marginal P(Y=y|x) and the
   conditional P(Y=y|x,z) are different numbers.

End your response with a line "FINAL_JSON:" followed by ONLY the JSON object below,
with no markdown formatting:
{{
  "TV": <float>,
  "TE": <float>,
  "DE": <float>,
  "IE": <float>
}}"""


###############################################################################
# Reference against model answer
###############################################################################

def compute_discrepancies(ground_truth: dict, llm_effects: dict) -> pd.DataFrame:
    # Only the five effects the LLM is asked for. "IE" here is the direct form of
    # Eq. 8; "IE_reverse" is in ground_truth but stays out of the table, since
    # comparing an answer against an estimand nobody requested measures nothing.
    rows = []
    for effect in ["TV", "TE", "SE", "DE", "IE"]:
        gt  = ground_truth.get(effect, float("nan"))
        llm_val = float(llm_effects.get(effect, float("nan")))
        abs_err = abs(gt - llm_val)
        rel_err = abs_err / abs(gt) if abs(gt) > 1e-9 else float("nan")
        rows.append({
            "effect":      effect,
            "fairmind":    round(gt,  6),
            "llm":         round(llm_val, 6),
            "abs_error":   round(abs_err, 6),
            "rel_error_%": round(rel_err * 100, 2) if not pd.isna(rel_err) else float("nan"),
        })
    return pd.DataFrame(rows)
