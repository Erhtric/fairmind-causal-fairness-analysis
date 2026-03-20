import json
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph
from openai import OpenAI
from pgmpy.estimators import BayesianEstimator

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.effects import (
    categorical_natural_direct_effect,
    categorical_natural_indirect_effect,
    categorical_total_effect,
    categorical_total_variation,
    decompose_indirect_effect,
    decompose_spurious_effect,
    natural_direct_effect,
    natural_indirect_effect,
    spurious_effect,
    total_effect,
    total_variation,
)
from src.graph import build_sfm
from src.llm import prepare_llm_payload_general, summarize_with_llm_combined
from src.model import fit_discrete_bayesian_model
from src.visualisation.graph import visualize_sfm

load_dotenv(override=True)


# -------------------------------------------------------------------
# App helpers
# -------------------------------------------------------------------

def init_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI()


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    from io import BytesIO

    bio = BytesIO(file_bytes)
    suffix = suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(bio)
    if suffix == ".tsv":
        return pd.read_csv(bio, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(bio)
    if suffix == ".json":
        return pd.read_json(bio)
    raise ValueError(f"Unsupported file type: {suffix}")


@st.cache_data(show_spinner=False)
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.replace(r"^[^A-Za-z0-9]+$", np.nan, regex=True, inplace=True)
    out.dropna(inplace=True)
    return out


@st.cache_data(show_spinner=False)
def fit_bn_cached(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    w_cols: tuple[str, ...],
    z_cols: tuple[str, ...],
    sorted_mediators: bool,
    sorted_confounders: bool,
):
    sfm = build_sfm(
        sensitive_attr=x_col,
        outcome_attr=y_col,
        confounder_attrs=list(z_cols),
        mediator_attrs=list(w_cols),
        sorted_confounders=sorted_confounders,
        sorted_mediators=sorted_mediators,
        latents=None,
    )
    bn = fit_discrete_bayesian_model(
        sfm=sfm,
        data=df,
        estimator_instance=(BayesianEstimator, {"prior_type": "BDeu"}),
    )
    return sfm, bn



def round_or_none(x: Any, nd: int = 6) -> Any:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x



def unique_states(df: pd.DataFrame, col: str) -> list[Any]:
    vals = df[col].dropna().tolist()
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return seen

#NOT USED
def make_matrix_df(res):
    if res is None:
        return pd.DataFrame()

    # dict-style
    if isinstance(res, dict):
        matrix = res.get("matrix")
        x0_states = res.get("x0_states")
        x1_states = res.get("x1_states")
    else:
        # object-style
        matrix = getattr(res, "matrix", None)
        x0_states = getattr(res, "x0_states", None)
        x1_states = getattr(res, "x1_states", None)

    if matrix is None:
        raise ValueError(f"make_matrix_df expected a matrix result, got: {res}")

    return pd.DataFrame(matrix, index=x0_states, columns=x1_states)




def build_scalar_results(
    bn,
    y_col: str,
    y_value: Any,
    x_col: str,
    x0: Any,
    x1: Any,
    include_decomposition: bool,
) -> dict[str, Any]:
    target = (y_col, y_value)
    out = {
        "tv": total_variation(bn, target, x_col, x0, x1),
        "te": total_effect(bn, target, x_col, x0, x1),
        "de": natural_direct_effect(bn, target, x_col, x0, x1),
        "ie": natural_indirect_effect(bn, target, x_col, x1, x0),
        "sex1": spurious_effect(bn, target, x_col, x1),
        "sex0": spurious_effect(bn, target, x_col, x0),
    }

    if include_decomposition:
        try:
            out["ie_decomposition"] = decompose_indirect_effect(
                bn=bn,
                target=target,
                private_attr=x_col,
                x0=x1,
                x1=x0,
            )
        except Exception as exc:
            out["ie_decomposition_error"] = str(exc)

        try:
            out["se_decomposition_x1"] = decompose_spurious_effect(
                bn=bn,
                target=target,
                private_attr=x_col,
                x=x1,
            )
        except Exception as exc:
            out["se_decomposition_x1_error"] = str(exc)

        try:
            out["se_decomposition_x0"] = decompose_spurious_effect(
                bn=bn,
                target=target,
                private_attr=x_col,
                x=x0,
            )
        except Exception as exc:
            out["se_decomposition_x0_error"] = str(exc)

    return out

def scalar_results_to_tree_effects(scalar_results: dict) -> dict:
    return {
        "total_variation": scalar_results.get("tv"),
        "total_effect": scalar_results.get("te"),
        "direct_effect": scalar_results.get("de"),
        "indirect_effect": scalar_results.get("ie"),
        "spurious_effect_x1": scalar_results.get("sex1"),
        "spurious_effect_x0": scalar_results.get("sex0"),
        "indirect_effect_decomposition": scalar_results.get("ie_decomposition", {}),
        "spurious_effect_decomposition_x1": scalar_results.get("se_decomposition_x1", {}),
        "spurious_effect_decomposition_x0": scalar_results.get("se_decomposition_x0", {}),
    }

def build_effect_tree(effects: dict) -> Digraph:
    dot = Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontsize="10")

    def fmt(label: str, key: str):
        val = effects.get(key)
        if val is None:
            return label
        return f"{label}\n({round_or_none(val, nd=5)})"

    dot.node("TV", fmt("TV", "total_variation"))
    dot.node("TE", fmt("TE", "total_effect"))
    dot.node("SEx1", fmt("SE(x1)", "spurious_effect_x1"))
    dot.node("SEx0", fmt("SE(x0)", "spurious_effect_x0"))

    dot.edge("TV", "TE")
    dot.edge("TV", "SEx0")
    dot.edge("TV", "SEx1")

    dot.node("DE", fmt("DE", "direct_effect"))
    dot.node("IE", fmt("IE", "indirect_effect"))
    dot.edge("TE", "DE")
    dot.edge("TE", "IE")

    indirect_decomp = effects.get("indirect_effect_decomposition", {})
    if isinstance(indirect_decomp, dict) and len(indirect_decomp) > 0:
        for i, (name, val) in enumerate(indirect_decomp.items()):
            node_id = f"IE_{i}"
            dot.node(node_id, f"{name}\n({round_or_none(val, nd=5)})")
            dot.edge("IE", node_id)

    spurious_decomp_x1 = effects.get("spurious_effect_decomposition_x1", {})
    if isinstance(spurious_decomp_x1, dict) and len(spurious_decomp_x1) > 0:
        for j, (name, val) in enumerate(spurious_decomp_x1.items()):
            node_id = f"SEx1_{j}"
            dot.node(node_id, f"{name}\n({round_or_none(val, nd=5)})")
            dot.edge("SEx1", node_id)

    spurious_decomp_x0 = effects.get("spurious_effect_decomposition_x0", {})
    if isinstance(spurious_decomp_x0, dict) and len(spurious_decomp_x0) > 0:
        for j, (name, val) in enumerate(spurious_decomp_x0.items()):
            node_id = f"SEx0_{j}"
            dot.node(node_id, f"{name}\n({round_or_none(val, nd=5)})")
            dot.edge("SEx0", node_id)

    return dot


def compute_all_categorical_results(
    bn,
    y_col: str,
    y_value: Any,
    x_col: str,
    ordered_states: list[Any],
) -> dict[str, Any]:
    target = (y_col, y_value)

    te = categorical_total_effect(bn, target, x_col, ordered_states, ordered_states)
    tv = categorical_total_variation(bn, target, x_col, ordered_states, ordered_states)
    de = categorical_natural_direct_effect(bn, target, x_col, ordered_states, ordered_states)
    ie = categorical_natural_indirect_effect(bn, target, x_col, ordered_states, ordered_states)
    sex1 = {
    "value": spurious_effect(bn, target, x_col, ordered_states[-1]),
    "decomposition": None,
        }
    sex0 = {
        "value": spurious_effect(bn, target, x_col, ordered_states[0]),
        "decomposition": None,
        }
    return {
        "te": te,
        "tv": tv,
        "de": de,
        "ie": ie,
        "sex1": sex1,
        "sex0": sex0,

    }


def render_decomposition_dict(title: str, data: dict[str, Any] | None) -> None:
    st.markdown(f"**{title}**")
    if not data:
        st.info("No decomposition available.")
        return
    rows = [{"component": k, "value": round_or_none(v)} for k, v in data.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)



def serialize_stepwise_dict(d: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"step": k, "value": float(v)} for k, v in d.items()]



def compute_interesting_thresholds(curve_df: pd.DataFrame) -> pd.DataFrame:
    out = curve_df.copy().sort_values("threshold").reset_index(drop=True)
    if out.empty:
        return out

    for col in ["tv", "te", "de", "ie"]:
        out[f"abs_{col}"] = out[col].abs()
        out[f"delta_{col}"] = out[col].diff().abs().fillna(0.0)

    out["score"] = (
        out[["abs_tv", "abs_te", "abs_de", "abs_ie"]].sum(axis=1)
        + out[["delta_tv", "delta_te", "delta_de", "delta_ie"]].sum(axis=1)
    )

    ranked = out.sort_values("score", ascending=False).head(8).copy()
    keep = ["threshold", "tv", "te", "de", "ie", "score"]
    return ranked[keep].round(6)



def make_threshold_dataset(
    df: pd.DataFrame,
    y_col: str,
    threshold: float,
    direction: str,
) -> tuple[pd.DataFrame, str, int]:
    out = df.copy()
    y_bin_col = "__Ybin__"

    if direction == "Y ≤ threshold":
        out[y_bin_col] = (out[y_col].astype(float) <= float(threshold)).astype(int)
    else:
        out[y_bin_col] = (out[y_col].astype(float) >= float(threshold)).astype(int)

    return out, y_bin_col, 1



def compute_continuous_threshold_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    w_cols: list[str],
    z_cols: list[str],
    x0: Any,
    x1: Any,
    thresholds: list[float],
    direction: str,
    sorted_mediators: bool,
    sorted_confounders: bool,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, thr in enumerate(thresholds, start=1):
        status.write(f"Threshold {i}/{len(thresholds)}: {thr:.6g}")
        df_thr, y_bin_col, y_target = make_threshold_dataset(df, y_col, thr, direction)
        _, bn = fit_bn_cached(
            df=df_thr,
            x_col=x_col,
            y_col=y_bin_col,
            w_cols=tuple(w_cols),
            z_cols=tuple(z_cols),
            sorted_mediators=sorted_mediators,
            sorted_confounders=sorted_confounders,
        )

        target = (y_bin_col, y_target)
        rows.append(
            {
                "threshold": float(thr),
                "tv": total_variation(bn, target, x_col, x0, x1),
                "te": total_effect(bn, target, x_col, x0, x1),
                "de": natural_direct_effect(bn, target, x_col, x0, x1),
                "ie": natural_indirect_effect(bn, target, x_col, x1, x0),
            }
        )
        progress.progress(i / len(thresholds))

    progress.empty()
    status.empty()
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# it works for continuous 
def build_primary_payload(
    uploaded_name: str,
    sfm,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    w_cols: list[str],
    z_cols: list[str],
    x0: Any,
    x1: Any,
    y_target: Any,
    scalar_results: dict[str, Any],
    all_results: dict[str, Any],
    use_ordered_x: bool,
    sorted_mediators: bool,
    sorted_confounders: bool,
    variable_notes: str,
) -> dict[str, Any]:
    x_states = unique_states(df, x_col)
    y_states = unique_states(df, y_col)
    state_names = {
        x_col: x_states,
        y_col: y_states,
        **{c: unique_states(df, c) for c in w_cols},
        **{c: unique_states(df, c) for c in z_cols},
    }

    return prepare_llm_payload_general(
        dataset_name=uploaded_name,
        X=x_col,
        Y=y_col,
        W=w_cols,
        Z=z_cols,
        x0=x0,
        x1=x1,
        y_target=y_target,
        results={
            "primary_pair": scalar_results,
            "matrices": {k: v.matrix.tolist() for k, v in all_results.items()},
        },
        stepwise_results={
            k: serialize_stepwise_dict(v.get_stepwise_effects())
            for k, v in all_results.items()
            if use_ordered_x
        },
        variable_metadata={"notes": variable_notes},
        state_names=state_names,
        graph_edges=list(sfm.edges()),
        checks={
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "sorted_mediators": bool(sorted_mediators),
            "sorted_confounders": bool(sorted_confounders),
            "ordered_x": bool(use_ordered_x),
            "y_mode": "categorical",
        },
        notes=[
            "Spurious effect is reported separately at x0 and x1 for the selected primary pair.",
            "Pairwise matrices are computed over all selected X states.",
        ],
    )



# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

def main() -> None:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="margin-bottom:0;color:">FairMind</h1>
            <h1 style="margin-top:0;">
                Causal Fairness Analysis with LLMs
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
            """
    Upload a dataset for **causal fairness analysis**, specify:

    - **X**: sensitive attribute (e.g. race, gender)
    - **Y**: outcome (e.g. income)
    - **W**: mediator(s) if any
    - **Z**: confounder(s) if any

    Given the specified **SFM causal graph**, the app fits a **discrete Bayesian network** to the data and then computes a decomposition of the **total variation** into:
    *total effect*, *indirect effect*, *direct effect*, and *spurious effect*.
    
    Finally, the LLM generates a **report** summarizing the main findings.
    """
        )


    client = init_client()

    uploaded = st.file_uploader(
        "Upload a dataset",
        type=["csv", "tsv", "xlsx", "xls", "json"],
    )
    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    preprocess_mode = st.radio(
     "What is the status of your dataset?",
    ("Processed", "Raw (removing NaN and invalid symbols)"),
        horizontal=True,
    )


    try:
        df = load_dataframe(uploaded.getvalue(), Path(uploaded.name).suffix)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    if preprocess_mode.startswith("Raw"):
        before = len(df)
        df = clean_dataframe(df)
        st.success(f"Cleaning complete. Removed {before - len(df)} rows.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)

    columns = list(df.columns)
    if len(columns) < 2:
        st.error("The dataset must contain at least two columns.")
        return

    st.subheader("1. Variable roles")
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X: sensitive attribute", options=columns)
        remaining_y = [c for c in columns if c != x_col]
        y_col = st.selectbox("Y: outcome", options=remaining_y)
    with c2:
        remaining_other = [c for c in columns if c not in {x_col, y_col}]
        w_cols = st.multiselect("W: mediators", options=remaining_other)
        z_cols = st.multiselect(
            "Z: confounders",
            options=[c for c in remaining_other if c not in set(w_cols)],
        )

    if set(w_cols) & set(z_cols):
        st.error("W and Z must be disjoint.")
        return

    st.subheader("2. Configuration of Outcome and Private Attribute")

    st.markdown("### Outcome configuration")

    y_mode = st.radio(
        "Y type",
        ["Categorical / discrete", "Continuous via threshold analysis"],
        horizontal=True,
        )
    if y_mode == "Categorical / discrete":
        y_states = unique_states(df, y_col)
        if len(y_states) == 0:
            st.error("Y has no observed states.")
            return
        y_value = st.selectbox("Target Y state", options=y_states, index=0)
    else:
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            st.error("Continuous threshold analysis requires a numeric Y column.")
            return

        s = df[y_col].dropna().astype(float)
        st.caption(f"Observed Y range: [{s.min():.6g}, {s.max():.6g}]")

        c5, c6, c7 = st.columns(3)
        with c5:
            threshold_direction = st.selectbox(
                "Threshold direction",
                ["Y ≤ threshold", "Y ≥ threshold"],
            )
        with c6:
            grid_kind = st.selectbox("Threshold grid", ["Quantiles", "Evenly spaced"])
        with c7:
            n_thresholds = st.slider("Number of thresholds", min_value=5, max_value=100, value=25)

        if grid_kind == "Quantiles":
            qs = np.linspace(0.01, 0.99, n_thresholds)
            thresholds = np.quantile(s.to_numpy(), qs)
        else:
            thresholds = np.linspace(float(s.min()), float(s.max()), n_thresholds)
        thresholds = np.unique(thresholds.astype(float)).tolist()

    st.markdown("### Private Attribute configuration")
    x_states = unique_states(df, x_col)
    if len(x_states) < 2:
        st.error("X must have at least two observed states.")
        return

    c3, c4 = st.columns(2)
    with c3:
        x0 = st.selectbox("x0", options=x_states, index=0)
    with c4:
        x1_candidates = [v for v in x_states if v != x0]
        x1 = st.selectbox("x1", options=x1_candidates, index=0)

    st.subheader("3. Optional ordering")
    use_ordered_x = st.toggle(
        "Treat X as ordered and compute stepwise effects",
        value=False,
    )

    ordered_x_states = x_states
    if use_ordered_x:
        ordered_x_states = st.multiselect(
            "Ordered X states",
            options=x_states,
            default=x_states,
            help="Select all X states in the intended order.",
        )
        if set(ordered_x_states) != set(x_states):
            st.error("To compute ordered effects, include all X states exactly once.")
            return

    sorted_mediators = False
    if len(w_cols) > 1:
        sorted_mediators = st.checkbox("Mediators are topologically ordered", value=False)

    sorted_confounders = False
    if len(z_cols) > 1:
        sorted_confounders = st.checkbox("Confounders are topologically ordered", value=False)

    include_decomposition = st.checkbox("Compute mediator/confounder decompositions", value=True)
    variable_notes = st.text_area(
        "Variable notes (optional)",
        placeholder="Describe the meanings of X, Y, W, Z and the target state of Y.",
    )


    if "analysis_ran" not in st.session_state:
        st.session_state.analysis_ran = False

    if st.button("Run analysis", type="primary"):
        st.session_state.analysis_ran = True

    if not st.session_state.analysis_ran:
        return

    if y_mode == "Categorical / discrete":
        try:
            with st.spinner("Building SFM and fitting Bayesian model..."):
                sfm, bn = fit_bn_cached(
                    df=df,
                    x_col=x_col,
                    y_col=y_col,
                    w_cols=tuple(w_cols),
                    z_cols=tuple(z_cols),
                    sorted_mediators=sorted_mediators,
                    sorted_confounders=sorted_confounders,
                )
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")
            return

        st.success("Model fitted successfully.")
        st.subheader("4. Causal graph (SFM)")
        fig = visualize_sfm(sfm)
        st.pyplot(fig)


        st.subheader("5. General Effects")
        all_results = compute_all_categorical_results(
                bn=bn,
                y_col=y_col,
                y_value=y_value,
                x_col=x_col,
                ordered_states=ordered_x_states if use_ordered_x else x_states,
            )
        scalar_results = build_scalar_results(
            bn=bn,
            y_col=y_col,
            y_value=y_value,
            x_col=x_col,
            x0=x0,
            x1=x1,
            include_decomposition=include_decomposition,
        )
        raw_rows = pd.DataFrame(
            [{"effect": k, "value": round_or_none(v)} for k, v in scalar_results.items() if not isinstance(v, dict)]
        )
        st.dataframe(raw_rows, use_container_width=True)
        st.markdown("**Effect decomposition tree**")

        tree_effects = scalar_results_to_tree_effects(scalar_results)
        st.graphviz_chart(build_effect_tree(tree_effects), use_container_width=True)
        if include_decomposition:
            c8, c9, c10 = st.columns(3)
            with c8:
                render_decomposition_dict(
                    "Indirect-effect decomposition",
                    scalar_results.get("ie_decomposition"),
                )
            with c9:
                render_decomposition_dict(
                    f"Spurious-effect decomposition at x1 = {x1}",
                    scalar_results.get("se_decomposition_x1"),
                )
            with c10:
                render_decomposition_dict(
                    f"Spurious-effect decomposition at x0 = {x0}",
                    scalar_results.get("se_decomposition_x0"),
                )
    
        # st.subheader("6. All pairwise effects across X states")
        # tabs = st.tabs([
        #     "Total Variation",
        #     "Total Effect",
        #     "Direct Effect",
        #     "Indirect Effect",
        #     "Spurious Effect",
        # ])

        # for tab, key, label in zip(
        #     tabs[:4],
        #     ["tv", "te", "de", "ie"],
        #     ["TV", "TE", "DE", "IE"],
        #     strict=False,
        # ):
        #     with tab:
        #         res = all_results[key]
        #         st.markdown(f"**{label} matrix**")
        #         st.dataframe(make_matrix_df(res), use_container_width=True)

        #         max_val, max_x0, max_x1 = res.max_disparity()
        #         st.caption(
        #             f"Max |{label}| at x0={max_x0}, x1={max_x1}: {round_or_none(max_val)}"
        #         )

        # with tabs[4]:
        #     sex0_val = all_results["sex0"]["value"]
        #     sex1_val = all_results["sex1"]["value"]

        #     x_states_used = ordered_x_states if use_ordered_x else x_states
        #     x0_state = x_states_used[0]
        #     x1_state = x_states_used[-1]

        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.markdown(f"**SE({x0_state})**: {round_or_none(sex0_val)}")
        #     with col2:
        #         st.markdown(f"**SE({x1_state})**: {round_or_none(sex1_val)}")
        if use_ordered_x:
            st.subheader("Stepwise effects")

            step_rows = []
            for i in range(len(ordered_x_states) - 1):
                x0_step = ordered_x_states[i]
                x1_step = ordered_x_states[i + 1]

                step_res = build_scalar_results(
                    bn=bn,
                    y_col=y_col,
                    y_value=y_value,
                    x_col=x_col,
                    x0=x0_step,
                    x1=x1_step,
                    include_decomposition=False,  # not needed for stepwise table
                )

                step_rows.append({
                    "step": f"{x0_step} -> {x1_step}",
                    "TV": round_or_none(step_res.get("tv")),
                    "TE": round_or_none(step_res.get("te")),
                    "DE": round_or_none(step_res.get("de")),
                    "IE": round_or_none(step_res.get("ie")),
                })
            step_df = pd.DataFrame(step_rows)
            st.dataframe(pd.DataFrame(step_df), use_container_width=True)
        
            # reversal_messages = []
            # for effect_name, effect_res in [
            #     ("TV", all_results["tv"]),
            #     ("TE", all_results["te"]),
            #     ("DE", all_results["de"]),
            #     ("IE", all_results["ie"]),
            # ]:
            #     reversals = effect_res.find_sign_reversals()
            #     if reversals:
            #         reversal_messages.extend([f"{effect_name}: {msg}" for msg in reversals])

            # if reversal_messages:
            #     st.warning("; ".join(reversal_messages))
            # else:
            #     st.success("No sign reversals detected in adjacent steps for TV, TE, DE, or IE.")
            st.markdown("**Ordered effect curve**")

            y_vals = [0.0]
            for val in step_df["TE"]:
                y_vals.append(y_vals[-1] + float(val))

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.step(range(len(ordered_x_states)), y_vals, where="mid")
            ax.set_xticks(range(len(ordered_x_states)))
            ax.set_xticklabels(ordered_x_states, rotation=20, ha="right")
            ax.set_ylabel("TE")
            ax.set_xlabel("Ordered X categories")
            ax.set_title(f"TE from {ordered_x_states[0]} across ordered X states")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        st.subheader("7. Exportable JSON payload")
        llm_payload = build_primary_payload(
            uploaded_name=uploaded.name,
            sfm=sfm,
            df=df,
            x_col=x_col,
            y_col=y_col,
            w_cols=w_cols,
            z_cols=z_cols,
            x0=x0,
            x1=x1,
            y_target=y_value,
            scalar_results=scalar_results,
            all_results=all_results,
            use_ordered_x=use_ordered_x,
            sorted_mediators=sorted_mediators,
            sorted_confounders=sorted_confounders,
            variable_notes=variable_notes,
        )

        payload_json = json.dumps(llm_payload, indent=2, ensure_ascii=False, default=str)
        st.code(payload_json, language="json")
        st.download_button(
            "Download JSON payload",
            data=payload_json.encode("utf-8"),
            file_name="fairmind_results.json",
            mime="application/json",
        )

        st.subheader("8. LLM report")
        if client is None:
            st.info("OPENAI_API_KEY not found. Set it in your environment to enable report generation.")
            return

        if st.button("Generate LLM report"):
            try:
                with st.spinner("Generating report..."):
                    text, latex_doc, token_usage = summarize_with_llm_combined(llm_payload, client)
            except Exception as exc:
                st.error(f"LLM report generation failed: {exc}")
                return

            st.session_state["llm_text"] = text
            st.session_state["llm_latex"] = latex_doc
            st.session_state["llm_token_usage"] = token_usage

    else:
        st.subheader("4. Continuous Y threshold exploration")
        st.info(
            "Y is converted into a binary event at each threshold, then TV, TE, DE and IE are recomputed. "
            "Use the step graph to spot interesting thresholds and then inspect one in detail."
        )

        try:
            curve_df = compute_continuous_threshold_curve(
                df=df,
                x_col=x_col,
                y_col=y_col,
                w_cols=w_cols,
                z_cols=z_cols,
                x0=x0,
                x1=x1,
                thresholds=thresholds,
                direction=threshold_direction,
                sorted_mediators=sorted_mediators,
                sorted_confounders=sorted_confounders,
            )
        except Exception as exc:
            st.error(f"Threshold analysis failed: {exc}")
            return

        chart_df = curve_df.set_index("threshold")[["tv", "te", "de", "ie"]]
        st.line_chart(chart_df)
        st.dataframe(curve_df.round(6), use_container_width=True)

        interesting_df = compute_interesting_thresholds(curve_df)
        st.markdown("**Suggested interesting thresholds**")
        st.caption("The score is higher when effects are large in magnitude or change sharply from nearby thresholds.")
        st.dataframe(interesting_df, use_container_width=True)

        default_thr = float(curve_df.iloc[len(curve_df) // 2]["threshold"])
        threshold_choice = st.select_slider(
            "Choose a threshold to inspect in detail",
            options=[float(x) for x in curve_df["threshold"].tolist()],
            value=default_thr,
        )

        df_selected, y_bin_col, y_target = make_threshold_dataset(
            df=df,
            y_col=y_col,
            threshold=threshold_choice,
            direction=threshold_direction,
        )

        try:
            with st.spinner("Fitting model at selected threshold..."):
                sfm, bn = fit_bn_cached(
                    df=df_selected,
                    x_col=x_col,
                    y_col=y_bin_col,
                    w_cols=tuple(w_cols),
                    z_cols=tuple(z_cols),
                    sorted_mediators=sorted_mediators,
                    sorted_confounders=sorted_confounders,
                )
                scalar_results = build_scalar_results(
                    bn=bn,
                    y_col=y_bin_col,
                    y_value=y_target,
                    x_col=x_col,
                    x0=x0,
                    x1=x1,
                    include_decomposition=include_decomposition,
                )
                if use_ordered_x:
                    st.subheader("Stepwise effects at selected threshold")

                    step_rows = []
                    for i in range(len(ordered_x_states) - 1):
                        x0_step = ordered_x_states[i]
                        x1_step = ordered_x_states[i + 1]

                        step_res = build_scalar_results(
                            bn=bn,
                            y_col=y_bin_col,
                            y_value=y_target,
                            x_col=x_col,
                            x0=x0_step,
                            x1=x1_step,
                            include_decomposition=False,
                        )

                        step_rows.append({
                            "step": f"{x0_step} -> {x1_step}",
                            "TV": round_or_none(step_res.get("tv")),
                            "TE": round_or_none(step_res.get("te")),
                            "DE": round_or_none(step_res.get("de")),
                            "IE": round_or_none(step_res.get("ie")),
                        })

                    step_df = pd.DataFrame(step_rows)
                    st.dataframe(step_df, use_container_width=True)

                    y_vals = [0.0]
                    for val in step_df["TE"]:
                        y_vals.append(y_vals[-1] + float(val))

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.step(range(len(ordered_x_states)), y_vals, where="mid")
                    ax.set_xticks(range(len(ordered_x_states)))
                    ax.set_xticklabels(ordered_x_states, rotation=20, ha="right")
                    ax.set_ylabel("TE")
                    ax.set_xlabel("Ordered X categories")
                    ax.set_title(f"TE from {ordered_x_states[0]} across ordered X states")
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
        except Exception as exc:
            st.error(f"Detailed threshold analysis failed: {exc}")
            return
        

        st.subheader("5. Selected-threshold results")
        st.write(
            f"Detailed results for threshold **{threshold_choice:.6g}** with event **{threshold_direction.replace('threshold', str(round(threshold_choice, 6)))}**."
        )
        fig = visualize_sfm(sfm)
        st.pyplot(fig)

        detail_rows = pd.DataFrame(
            [{"effect": k, "value": round_or_none(v)} for k, v in scalar_results.items() if not isinstance(v, dict)]
        )
        st.dataframe(detail_rows, use_container_width=True)

        if include_decomposition:
            c10, c11, c12 = st.columns(3)
            with c10:
                render_decomposition_dict(
                    "Indirect-effect decomposition",
                    scalar_results.get("ie_decomposition")
                )
            with c11:
                render_decomposition_dict(
                    "Spurious-effect decomposition at x1",
                    scalar_results.get("se_decomposition_x1")
                )
            with c12:
                render_decomposition_dict(
                    "Spurious-effect decomposition at x0",
                    scalar_results.get("se_decomposition_x0")
                )
        st.subheader("6. LLM payload for selected threshold")
        llm_payload = {
            "analysis_type": "continuous_threshold",
            "dataset_name": uploaded.name,
            "X": x_col,
            "Y": y_col,
            "W": w_cols,
            "Z": z_cols,
            "x0": x0,
            "x1": x1,
            "selected_threshold": float(threshold_choice),
            "threshold_direction": threshold_direction,
            "target_event": {y_bin_col: y_target},
            "curve": curve_df.round(6).to_dict(orient="records"),
            "interesting_thresholds": interesting_df.to_dict(orient="records"),
            "selected_threshold_results": scalar_results,
            "variable_metadata": {"notes": variable_notes},
            "graph_edges": list(sfm.edges()),
            "checks": {
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "sorted_mediators": bool(sorted_mediators),
                "sorted_confounders": bool(sorted_confounders),
                "y_mode": "continuous_threshold",
            },
            "notes": [
                "Each threshold creates a binary target and refits the discrete Bayesian network.",
                "The selected threshold is the one inspected in detail below.",
            ],
        }

        payload_json = json.dumps(llm_payload, indent=2, ensure_ascii=False, default=str)
        #st.code(payload_json, language="json")
        st.download_button(
            "Download JSON payload",
            data=payload_json.encode("utf-8"),
            file_name="fairmind_continuous_threshold_results.json",
            mime="application/json",
        )

        st.subheader("7. LLM report")
        if client is None:
            st.info("OPENAI_API_KEY not found. Set it in your environment to enable report generation.")
            return

        if st.button("Generate LLM report"):
            try:
                with st.spinner("Generating report..."):
                    text, latex_doc, token_usage = summarize_with_llm_combined(llm_payload, client)
            except Exception as exc:
                st.error(f"LLM report generation failed: {exc}")
                return

            st.session_state["llm_text"] = text
            st.session_state["llm_latex"] = latex_doc
            st.session_state["llm_token_usage"] = token_usage

    if st.session_state.get("llm_text"):
        st.markdown(st.session_state["llm_text"])

    if st.session_state.get("llm_latex"):
        with st.expander("LaTeX source"):
            st.code(st.session_state["llm_latex"], language="latex")

        st.download_button(
            "Download LaTeX report",
            data=st.session_state["llm_latex"].encode("utf-8"),
            file_name="fairmind_report.tex",
            mime="application/x-tex",
        )

    if st.session_state.get("llm_token_usage") is not None:
        st.markdown("**Token usage**")
        st.write(st.session_state["llm_token_usage"])


if __name__ == "__main__":
    main()
