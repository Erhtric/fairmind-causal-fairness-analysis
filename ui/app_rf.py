"""FairMind Streamlit app.

Run with: (uv run) streamlit run ui/app_rf.py
"""

import json
import os
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from graphviz import Digraph
from openai import OpenAI

# pgmpy >= 1.1 moved the discrete estimators
try:
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator
except ImportError:
    from pgmpy.estimators import DiscreteBayesianEstimator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-5.4-nano"
FAIRMIND_PROMPT = "fairmind_v2.txt"

from src.effects import (  # noqa
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
from src.graph import build_sfm, filter_nodes_by_type  # noqa: E402
from src.llm import prepare_llm_payload_general, summarize_fairmind  # noqa: E402
from src.model import fit_discrete_bayesian_model  # noqa: E402
from src.visualisation.graph import visualize_sfm  # noqa: E402

load_dotenv(override=True)


# -------------------------------------------------------------------
# Configuration objects
# -------------------------------------------------------------------


@dataclass(frozen=True)
class RoleConfig:
    """Variable-role assignment: sensitive attribute, outcome, mediators, confounders."""

    x_col: str
    y_col: str
    w_cols: tuple[str, ...]
    z_cols: tuple[str, ...]

    @property
    def sorted_mediators(self) -> bool:
        return len(self.w_cols) > 1

    @property
    def sorted_confounders(self) -> bool:
        return len(self.z_cols) > 1


@dataclass(frozen=True)
class XPairConfig:
    """Primary x0/x1 comparison pair and optional ordering of X states."""

    x0: Any
    x1: Any
    use_ordered_x: bool
    ordered_x_states: tuple[Any, ...]


@dataclass(frozen=True)
class OutcomeConfig:
    """Outcome handling: a categorical target state, or a continuous threshold sweep."""

    mode: str  # "categorical" | "threshold"
    y_value: Any = None
    thresholds: tuple[float, ...] = ()
    direction: str = ""


@dataclass(frozen=True)
class LLMConfig:
    """Settings for an OpenAI-compatible provider used for report generation."""

    api_key: str
    base_url: str
    model: str
    effort: str  # reasoning effort: "low" | "medium" | "high"

    def make_client(self) -> OpenAI | None:
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url or None)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def render_llm_provider_sidebar() -> LLMConfig:
    """Sidebar panel where the user configures their own OpenAI-compatible provider."""
    with st.sidebar:
        st.header("LLM provider")
        st.caption(
            "Bring your own OpenAI-compatible provider. It must support the "
            "Responses API, which is used for report generation."
        )
        base_url = st.text_input(
            "Base URL",
            value=os.getenv("OPENAI_BASE_URL", DEFAULT_LLM_BASE_URL),
            help="The provider's OpenAI-compatible endpoint.",
        )
        api_key = st.text_input(
            "API key",
            value="",
            type="password",
            help=(
                "Kept only in this browser session, never stored. "
                "Leave empty to use the OPENAI_API_KEY environment variable."
            ),
        )
        model = st.text_input(
            "Model",
            value=os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL),
            help="Model name as known by the provider.",
        )
        effort = st.selectbox(
            "Reasoning effort",
            options=["low", "medium", "high"],
            index=2,
            help="Passed to reasoning models; some providers/models may not support it.",
        )

    return LLMConfig(
        api_key=api_key.strip() or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url.strip(),
        model=model.strip(),
        effort=effort,
    )


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, suffix: str) -> pd.DataFrame:
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
        estimator_instance=(DiscreteBayesianEstimator, {"prior_type": "BDeu"}),
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


def make_matrix_df(res):
    if res is None:
        return None

    if isinstance(res, dict):
        matrix = res.get("matrix")
        x0_states = res.get("x0_states")
        x1_states = res.get("x1_states")
        mediators = res.get("mediators", None)  # optional
    else:
        matrix = getattr(res, "matrix", None)
        x0_states = getattr(res, "x0_states", None)
        x1_states = getattr(res, "x1_states", None)
        mediators = getattr(res, "mediators", None)

    if matrix is None:
        raise ValueError(f"make_matrix_df expected a matrix result, got: {res}")

    arr = np.asarray(matrix)

    if arr.ndim == 2:
        return [("", pd.DataFrame(arr, index=x0_states, columns=x1_states))]

    elif arr.ndim == 3:
        slices = []

        total = np.zeros((len(x0_states), len(x1_states)))

        for i in range(arr.shape[2]):
            name = (
                mediators[i]
                if mediators is not None and i < len(mediators)
                else f"Mediator {i}"
            )

            slice_matrix = arr[:, :, i]
            total += slice_matrix
            df = pd.DataFrame(slice_matrix, index=x0_states, columns=x1_states)
            slices.append((name, df))
        total_df = pd.DataFrame(total, index=x0_states, columns=x1_states)
        slices.append(("Total (sum of mediators)", total_df))
        return slices

    else:
        raise ValueError(f"Unsupported matrix shape: {arr.shape}")


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
        "spurious_effect_decomposition_x1": scalar_results.get(
            "se_decomposition_x1", {}
        ),
        "spurious_effect_decomposition_x0": scalar_results.get(
            "se_decomposition_x0", {}
        ),
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

    # The categorical DE/IE identification formulas need at least one mediator;
    # without one, DE would equal TE and IE would be zero, so skip them instead.
    has_mediators = bool(
        filter_nodes_by_type(bn.nodes(data=True, default={}), type="mediator")
    )
    de = ie = None
    if has_mediators:
        de = categorical_natural_direct_effect(
            bn, target, x_col, ordered_states, ordered_states
        )
        ie = categorical_natural_indirect_effect(
            bn, target, x_col, ordered_states, ordered_states
        )
    return {
        "te": te,
        "tv": tv,
        "de": de,
        "ie": ie,
    }


def serialize_stepwise_dict(d: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"step": k, "value": float(v)} for k, v in d.items()]


def compute_interesting_thresholds(curve_df: pd.DataFrame) -> pd.DataFrame:
    out = curve_df.copy().sort_values("threshold").reset_index(drop=True)
    if out.empty:
        return out

    for col in ["tv", "te", "de", "ie"]:
        out[f"abs_{col}"] = out[col].abs()
        out[f"delta_{col}"] = out[col].diff().abs().fillna(0.0)

    out["score"] = out[["abs_tv", "abs_te", "abs_de", "abs_ie"]].sum(axis=1) + out[
        ["delta_tv", "delta_te", "delta_de", "delta_ie"]
    ].sum(axis=1)

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


def reset_analysis_state():
    st.session_state.analysis_ran = False
    st.session_state.pop("results", None)
    st.session_state.pop("report", None)
    st.session_state.pop("effect_table", None)
    st.session_state.pop("report_csv_frames", None)


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
    all_results: dict[str, Any] | None,
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
    all_results = all_results or {}

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
            "matrices": {
                k: v.matrix.tolist() for k, v in all_results.items() if v is not None
            },
        },
        stepwise_results={
            k: serialize_stepwise_dict(v.get_stepwise_effects())
            for k, v in all_results.items()
            if use_ordered_x and v is not None
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
# Shared render components
# -------------------------------------------------------------------


def render_decomposition_dict(title: str, data: dict[str, Any] | None) -> None:
    st.markdown(f"**{title}**")
    if not data:
        st.info("No decomposition available.")
        return
    rows = [{"component": k, "value": round_or_none(v)} for k, v in data.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render_scalar_results_table(scalar_results: dict[str, Any]) -> None:
    rows = pd.DataFrame(
        [
            {"effect": k, "value": round_or_none(v)}
            for k, v in scalar_results.items()
            if not isinstance(v, dict)
        ]
    )
    st.dataframe(rows, width="stretch")


def render_decomposition_columns(
    scalar_results: dict[str, Any], x0: Any, x1: Any
) -> None:
    c_ie, c_se1, c_se0 = st.columns(3)
    with c_ie:
        render_decomposition_dict(
            "Indirect-effect decomposition",
            scalar_results.get("ie_decomposition"),
        )
    with c_se1:
        render_decomposition_dict(
            f"Spurious-effect decomposition at x1 = {x1}",
            scalar_results.get("se_decomposition_x1"),
        )
    with c_se0:
        render_decomposition_dict(
            f"Spurious-effect decomposition at x0 = {x0}",
            scalar_results.get("se_decomposition_x0"),
        )


def render_stepwise_section(
    bn,
    y_col: str,
    y_value: Any,
    x_col: str,
    ordered_x_states: list[Any],
    title: str = "Stepwise effects",
) -> pd.DataFrame:
    st.subheader(title)

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

        step_rows.append(
            {
                "step": f"{x0_step} -> {x1_step}",
                "TV": round_or_none(step_res.get("tv")),
                "TE": round_or_none(step_res.get("te")),
                "DE": round_or_none(step_res.get("de")),
                "IE": round_or_none(step_res.get("ie")),
            }
        )

    step_df = pd.DataFrame(step_rows)
    st.dataframe(step_df, width="stretch")

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

    return step_df


def render_json_download(payload: dict[str, Any], file_name: str) -> None:
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        "Download JSON payload",
        data=payload_json.encode("utf-8"),
        file_name=file_name,
        mime="application/json",
    )


def _safe_csv_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def scalar_results_to_frames(
    scalar_results: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    frames = {
        "scalar_effects": pd.DataFrame(
            [
                {"effect": k, "value": round_or_none(v)}
                for k, v in scalar_results.items()
                if not isinstance(v, dict)
            ]
        )
    }
    for key in ("ie_decomposition", "se_decomposition_x1", "se_decomposition_x0"):
        data = scalar_results.get(key)
        if isinstance(data, dict) and data:
            frames[key] = pd.DataFrame(
                [{"component": k, "value": round_or_none(v)} for k, v in data.items()]
            )
    return frames


def matrix_results_to_frames(all_results: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames = {}
    for key, res in all_results.items():
        if res is None:
            continue
        for name, matrix_df in make_matrix_df(res):
            suffix = f"_{_safe_csv_name(name)}" if name else ""
            frames[f"matrix_{key}{suffix}"] = matrix_df
    return frames


def build_report_zip(
    text: str, latex: str, csv_frames: dict[str, pd.DataFrame] | None
) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.md", text)
        zf.writestr("report.tex", latex)
        for name, frame in (csv_frames or {}).items():
            # matrices are indexed by the x0 states; plain tables carry no index
            with_index = name.startswith("matrix_")
            zf.writestr(
                f"csv/{name}.csv",
                frame.to_csv(
                    index=with_index, index_label="x0" if with_index else None
                ),
            )
    return buf.getvalue()


def render_llm_report_section(
    llm: LLMConfig, llm_payload: dict[str, Any], title: str
) -> None:
    st.subheader(title)
    client = llm.make_client()
    if client is None:
        st.info(
            "No API key configured. Enter one in the sidebar (LLM provider) "
            "or set the OPENAI_API_KEY environment variable to enable report generation."
        )
        return
    if not llm.model:
        st.info("No model configured. Set a model name in the sidebar (LLM provider).")
        return

    st.caption(f"Report will be generated by `{llm.model}` at `{llm.base_url}`.")
    if st.button("Generate LLM report"):
        try:
            with st.spinner("Generating report..."):
                text, latex_doc, token_usage = summarize_fairmind(
                    llm_payload,
                    client,
                    model=llm.model,
                    prompt_path=FAIRMIND_PROMPT,
                    effort=llm.effort,
                )
        except Exception as exc:
            st.error(f"LLM report generation failed: {exc}")
            return

        st.session_state["llm_text"] = text
        st.session_state["llm_latex"] = latex_doc
        st.session_state["llm_token_usage"] = token_usage


def render_llm_output_from_state() -> None:
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

    if st.session_state.get("llm_text") and st.session_state.get("llm_latex"):
        st.download_button(
            "Download report (text + LaTeX + CSVs)",
            data=build_report_zip(
                st.session_state["llm_text"],
                st.session_state["llm_latex"],
                st.session_state.get("report_csv_frames"),
            ),
            file_name="fairmind_report.zip",
            mime="application/zip",
            help="ZIP with the text report, the LaTeX source, and every numerical table as CSV.",
        )

    if st.session_state.get("llm_token_usage") is not None:
        st.markdown("**Token usage**")
        st.write(st.session_state["llm_token_usage"])


# -------------------------------------------------------------------
# Page sections (input widgets)
# -------------------------------------------------------------------


def render_intro() -> None:
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
    **USAGE**: Upload a dataset for **causal fairness analysis**. Then specify:

    - **X**: sensitive attribute (e.g. race, gender)
    - **Y**: outcome (e.g. income)
    - **W**: mediator(s) if any
    - **Z**: confounder(s) if any

    Following the **SFM** causal graph specification (see [Plecko D. & Bareinboim E., 2023](https://arxiv.org/abs/2207.11385)) over the
    specified partition of variables, the app fits a **discrete Bayesian network** (using [`pgmpy`](https://pgmpy.org/)) to the data and then computes a decomposition of the 
    **total variation** into: *total effect*, *indirect effect*, *direct effect*, and *spurious effect*.

    Finally, on user request, the LLM generates a **report** summarising the main findings. In order
    to do so, the application requires an **API key** to an OpenAI-compatible provider that can be
    inserted in the left sidebar, along with the **model** to use and the thinking effort.

    The report is generated in **text** and **LaTeX** formats, and can be downloaded along with all numerical tables as CSV files.
    """
    )
    st.info(
        """
    This application was developed as part of the research presented in the paper
    *“Automatic Causal Fairness Analysis with LLM-Generated Reporting.”*

    For questions or further information, please contact:
    **alessia.berarducci@supsi.ch**, **eric.rossetto@supsi.ch**, **alessandro.antonucci@supsi.ch**, **marco.zaffalon@supsi.ch**
    """
    )


def render_discretization(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Preprocessing: discretize numeric variables")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_columns:
        st.caption("No numeric columns available for discretization.")
        return df

    enable_discretization = st.checkbox(
        "Group numeric variables into categories before analysis",
        value=False,
        help="Useful for numeric variables with many distinct values.",
        on_change=reset_analysis_state,
    )

    if not enable_discretization:
        return df

    vars_to_discretize = st.multiselect(
        "Select numeric variables to group",
        options=numeric_columns,
        on_change=reset_analysis_state,
    )

    for col in vars_to_discretize:
        st.markdown(f"#### Discretization for `{col}`")

        method = st.radio(
            f"Method for {col}",
            options=["Equal width", "Quantile"],
            key=f"method_{col}",
            horizontal=True,
            on_change=reset_analysis_state,
        )

        new_col_name = st.text_input(
            f"New column name for {col}",
            value=f"{col}_group",
            key=f"name_{col}",
            on_change=reset_analysis_state,
        )

        try:
            if method == "Equal width":
                bins = st.slider(
                    f"Number of bins for {col}",
                    min_value=2,
                    max_value=10,
                    value=4,
                    key=f"bins_{col}",
                    on_change=reset_analysis_state,
                )

                df[new_col_name] = pd.cut(
                    df[col],
                    bins=bins,
                    include_lowest=True,
                ).astype(str)

            else:
                q = st.slider(
                    f"Number of quantile groups for {col}",
                    min_value=2,
                    max_value=10,
                    value=4,
                    key=f"q_{col}",
                )

                # duplicates='drop' avoids errors when not enough unique values
                df[new_col_name] = pd.qcut(
                    df[col],
                    q=q,
                    duplicates="drop",
                ).astype(str)

            st.success(f"Created grouped variable: `{new_col_name}`")

        except Exception as exc:
            st.error(f"Could not discretize `{col}`: {exc}")

    return df


def render_upload_and_preprocess() -> tuple[pd.DataFrame, str] | None:
    uploaded = st.file_uploader(
        "Upload a dataset",
        type=["csv", "tsv", "xlsx", "xls", "json"],
    )
    if uploaded is None:
        st.info("Upload a file to begin.")
        return None

    preprocess_mode = st.radio(
        "What is the status of your dataset?",
        ("Processed", "Raw (removing NaN and invalid symbols)"),
        horizontal=True,
        on_change=reset_analysis_state,
    )

    try:
        df = load_dataframe(uploaded.getvalue(), Path(uploaded.name).suffix)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None

    if preprocess_mode.startswith("Raw"):
        before = len(df)
        df = clean_dataframe(df)
        st.success(f"Cleaning complete. Removed {before - len(df)} rows.")

    df = render_discretization(df)
    return df, uploaded.name


def render_role_selection(df: pd.DataFrame) -> RoleConfig | None:
    columns = list(df.columns)

    st.subheader("1. Variable roles")
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X: sensitive attribute", options=columns)
        remaining_y = [c for c in columns if c != x_col]
        y_col = st.selectbox("Y: outcome", options=remaining_y)
    with c2:
        remaining_other = [c for c in columns if c not in {x_col, y_col}]

        w_cols = st.multiselect(
            "W: mediators (select in topological order)",
            options=remaining_other,
            help="Select mediators from upstream to downstream in the causal graph.",
        )

        if w_cols:
            st.info("Mediator order: " + " → ".join(w_cols))

        z_cols = st.multiselect(
            "Z: confounders (select in topological order)",
            options=[c for c in remaining_other if c not in set(w_cols)],
            help="Select confounders in topological order.",
        )

        if z_cols:
            st.info("Confounder order: " + " → ".join(z_cols))

    if set(w_cols) & set(z_cols):
        st.error("W and Z must be disjoint.")
        return None

    return RoleConfig(
        x_col=x_col, y_col=y_col, w_cols=tuple(w_cols), z_cols=tuple(z_cols)
    )


def render_outcome_config(df: pd.DataFrame, roles: RoleConfig) -> OutcomeConfig | None:
    st.subheader("2. Configuration of Outcome and Private Attribute")

    st.markdown("### Outcome configuration")

    y_mode = st.radio(
        "Y type",
        ["Categorical / discrete", "Continuous via threshold analysis"],
        horizontal=True,
    )
    if y_mode == "Categorical / discrete":
        y_states = unique_states(df, roles.y_col)
        if len(y_states) == 0:
            st.error("Y has no observed states.")
            return None
        y_value = st.selectbox("Target Y state", options=y_states, index=0)
        return OutcomeConfig(mode="categorical", y_value=y_value)

    if not pd.api.types.is_numeric_dtype(df[roles.y_col]):
        st.error("Continuous threshold analysis requires a numeric Y column.")
        return None

    s = df[roles.y_col].dropna().astype(float)
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
        n_thresholds = st.slider(
            "Number of thresholds", min_value=5, max_value=100, value=25
        )

    if grid_kind == "Quantiles":
        qs = np.linspace(0.01, 0.99, n_thresholds)
        thresholds = np.quantile(s.to_numpy(), qs)
    else:
        thresholds = np.linspace(float(s.min()), float(s.max()), n_thresholds)
    thresholds = np.unique(thresholds.astype(float)).tolist()

    return OutcomeConfig(
        mode="threshold",
        thresholds=tuple(thresholds),
        direction=threshold_direction,
    )


def render_pair_config(df: pd.DataFrame, roles: RoleConfig) -> XPairConfig | None:
    st.markdown("### Private Attribute configuration")
    x_states = unique_states(df, roles.x_col)
    if len(x_states) < 2:
        st.error("X must have at least two observed states.")
        return None

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
            return None

    return XPairConfig(
        x0=x0,
        x1=x1,
        use_ordered_x=use_ordered_x,
        ordered_x_states=tuple(ordered_x_states),
    )


def render_analysis_options() -> tuple[bool, str]:
    include_decomposition = st.checkbox(
        "Compute mediator/confounder decompositions", value=True
    )
    variable_notes = st.text_area(
        "Variable notes (optional)",
        placeholder="Describe the meanings of X, Y, W, Z and the target state of Y. Handled as free-text context for the LLM report only.",
    )
    return include_decomposition, variable_notes


# -------------------------------------------------------------------
# Page flows
# -------------------------------------------------------------------


def run_categorical_flow(
    df: pd.DataFrame,
    uploaded_name: str,
    roles: RoleConfig,
    pair: XPairConfig,
    y_value: Any,
    include_decomposition: bool,
    variable_notes: str,
    llm: LLMConfig,
) -> None:
    try:
        with st.spinner("Building SFM and fitting Bayesian model..."):
            sfm, bn = fit_bn_cached(
                df=df,
                x_col=roles.x_col,
                y_col=roles.y_col,
                w_cols=roles.w_cols,
                z_cols=roles.z_cols,
                sorted_mediators=roles.sorted_mediators,
                sorted_confounders=roles.sorted_confounders,
            )
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}")
        return

    st.success("Model fitted successfully.")
    st.subheader("4. Causal graph (SFM)")
    fig = visualize_sfm(sfm)
    st.pyplot(fig, width="content")

    st.subheader("5. General Effects")

    scalar_results = build_scalar_results(
        bn=bn,
        y_col=roles.y_col,
        y_value=y_value,
        x_col=roles.x_col,
        x0=pair.x0,
        x1=pair.x1,
        include_decomposition=include_decomposition,
    )
    render_scalar_results_table(scalar_results)

    st.markdown("**Effect decomposition tree**")
    tree_effects = scalar_results_to_tree_effects(scalar_results)
    st.graphviz_chart(build_effect_tree(tree_effects), width="stretch")

    if include_decomposition:
        render_decomposition_columns(scalar_results, pair.x0, pair.x1)

    st.subheader("6. All pairwise effects across X states")
    all_results = compute_all_categorical_results(
        bn=bn,
        y_col=roles.y_col,
        y_value=y_value,
        x_col=roles.x_col,
        ordered_states=list(pair.ordered_x_states),
    )

    tabs = st.tabs(
        [
            "Total Variation",
            "Total Effect",
            "Direct Effect",
            "Indirect Effect",
        ]
    )

    for tab, key, label in zip(
        tabs,
        ["tv", "te", "de", "ie"],
        ["TV", "TE", "DE", "IE"],
        strict=False,
    ):
        with tab:
            res = all_results[key]
            if res is None:
                st.info(
                    f"{label} requires at least one mediator (W); none was selected."
                )
                continue
            st.write(f"**{label} matrix**")
            dfs = make_matrix_df(res)
            for name, df1 in dfs:
                st.markdown(f"**{name}**")
                st.dataframe(df1, width="stretch")
            max_val, max_x0, max_x1 = res.max_disparity()
            st.caption(
                f"Max |{label}| at x0={max_x0}, x1={max_x1}: {round_or_none(max_val)}"
            )

    step_df = None
    if pair.use_ordered_x:
        step_df = render_stepwise_section(
            bn=bn,
            y_col=roles.y_col,
            y_value=y_value,
            x_col=roles.x_col,
            ordered_x_states=list(pair.ordered_x_states),
        )

    csv_frames = {
        **scalar_results_to_frames(scalar_results),
        **matrix_results_to_frames(all_results),
    }
    if step_df is not None:
        csv_frames["stepwise_effects"] = step_df
    st.session_state["report_csv_frames"] = csv_frames

    st.subheader("7. Exportable JSON results")
    llm_payload = build_primary_payload(
        uploaded_name=uploaded_name,
        sfm=sfm,
        df=df,
        x_col=roles.x_col,
        y_col=roles.y_col,
        w_cols=list(roles.w_cols),
        z_cols=list(roles.z_cols),
        x0=pair.x0,
        x1=pair.x1,
        y_target=y_value,
        scalar_results=scalar_results,
        all_results=all_results,
        use_ordered_x=pair.use_ordered_x,
        sorted_mediators=roles.sorted_mediators,
        sorted_confounders=roles.sorted_confounders,
        variable_notes=variable_notes,
    )
    render_json_download(llm_payload, "fairmind_results.json")

    render_llm_report_section(llm, llm_payload, "8. LLM report")


def run_threshold_flow(
    df: pd.DataFrame,
    uploaded_name: str,
    roles: RoleConfig,
    pair: XPairConfig,
    outcome: OutcomeConfig,
    include_decomposition: bool,
    variable_notes: str,
    llm: LLMConfig,
) -> None:
    st.subheader("4. Continuous Y threshold exploration")
    st.info(
        "Y is converted into a binary event at each threshold, then TV, TE, DE and IE are recomputed. "
        "Use the step graph to spot interesting thresholds and then inspect one in detail."
    )

    try:
        curve_df = compute_continuous_threshold_curve(
            df=df,
            x_col=roles.x_col,
            y_col=roles.y_col,
            w_cols=list(roles.w_cols),
            z_cols=list(roles.z_cols),
            x0=pair.x0,
            x1=pair.x1,
            thresholds=list(outcome.thresholds),
            direction=outcome.direction,
            sorted_mediators=roles.sorted_mediators,
            sorted_confounders=roles.sorted_confounders,
        )
    except Exception as exc:
        st.error(f"Threshold analysis failed: {exc}")
        return

    chart_df = curve_df.set_index("threshold")[["tv", "te", "de", "ie"]]
    st.line_chart(chart_df)
    st.dataframe(curve_df.round(6), width="stretch")

    interesting_df = compute_interesting_thresholds(curve_df)
    st.markdown("**Suggested interesting thresholds**")
    st.caption(
        "The score is higher when effects are large in magnitude or change sharply from nearby thresholds."
    )
    st.dataframe(interesting_df, width="stretch")

    default_thr = float(curve_df.iloc[len(curve_df) // 2]["threshold"])
    threshold_choice = st.select_slider(
        "Choose a threshold to inspect in detail",
        options=[float(x) for x in curve_df["threshold"].tolist()],
        value=default_thr,
    )

    df_selected, y_bin_col, y_target = make_threshold_dataset(
        df=df,
        y_col=roles.y_col,
        threshold=threshold_choice,
        direction=outcome.direction,
    )

    try:
        with st.spinner("Fitting model at selected threshold..."):
            sfm, bn = fit_bn_cached(
                df=df_selected,
                x_col=roles.x_col,
                y_col=y_bin_col,
                w_cols=roles.w_cols,
                z_cols=roles.z_cols,
                sorted_mediators=roles.sorted_mediators,
                sorted_confounders=roles.sorted_confounders,
            )
            scalar_results = build_scalar_results(
                bn=bn,
                y_col=y_bin_col,
                y_value=y_target,
                x_col=roles.x_col,
                x0=pair.x0,
                x1=pair.x1,
                include_decomposition=include_decomposition,
            )
        step_df = None
        if pair.use_ordered_x:
            step_df = render_stepwise_section(
                bn=bn,
                y_col=y_bin_col,
                y_value=y_target,
                x_col=roles.x_col,
                ordered_x_states=list(pair.ordered_x_states),
                title="Stepwise effects at selected threshold",
            )
    except Exception as exc:
        st.error(f"Detailed threshold analysis failed: {exc}")
        return

    st.subheader("5. Selected-threshold results")
    st.write(
        f"Detailed results for threshold **{threshold_choice:.6g}** with event **{outcome.direction.replace('threshold', str(round(threshold_choice, 6)))}**."
    )
    fig = visualize_sfm(sfm)
    st.pyplot(fig, width="content")

    render_scalar_results_table(scalar_results)

    if include_decomposition:
        render_decomposition_columns(scalar_results, pair.x0, pair.x1)

    st.subheader("6. LLM payload for selected threshold")
    llm_payload = {
        "analysis_type": "continuous_threshold",
        "dataset_name": uploaded_name,
        "X": roles.x_col,
        "Y": roles.y_col,
        "W": list(roles.w_cols),
        "Z": list(roles.z_cols),
        "x0": pair.x0,
        "x1": pair.x1,
        "selected_threshold": float(threshold_choice),
        "threshold_direction": outcome.direction,
        "target_event": {y_bin_col: y_target},
        "curve": curve_df.round(6).to_dict(orient="records"),
        "interesting_thresholds": interesting_df.to_dict(orient="records"),
        "selected_threshold_results": scalar_results,
        "variable_metadata": {"notes": variable_notes},
        "graph_edges": list(sfm.edges()),
        "checks": {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "sorted_mediators": bool(roles.sorted_mediators),
            "sorted_confounders": bool(roles.sorted_confounders),
            "y_mode": "continuous_threshold",
        },
        "notes": [
            "Each threshold creates a binary target and refits the discrete Bayesian network.",
            "The selected threshold is the one inspected in detail below.",
        ],
    }
    render_json_download(llm_payload, "fairmind_continuous_threshold_results.json")

    csv_frames = {
        **scalar_results_to_frames(scalar_results),
        "threshold_curve": curve_df.round(6),
        "interesting_thresholds": interesting_df,
    }
    if step_df is not None:
        csv_frames["stepwise_effects"] = step_df
    st.session_state["report_csv_frames"] = csv_frames

    render_llm_report_section(llm, llm_payload, "7. LLM report")


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------


def main() -> None:
    render_intro()

    llm = render_llm_provider_sidebar()

    loaded = render_upload_and_preprocess()
    if loaded is None:
        return
    df, uploaded_name = loaded

    st.subheader("Dataset preview")
    st.dataframe(df.head(), width="stretch")

    if len(df.columns) < 2:
        st.error("The dataset must contain at least two columns.")
        return

    roles = render_role_selection(df)
    if roles is None:
        return

    outcome = render_outcome_config(df, roles)
    if outcome is None:
        return

    pair = render_pair_config(df, roles)
    if pair is None:
        return

    include_decomposition, variable_notes = render_analysis_options()

    if "analysis_ran" not in st.session_state:
        st.session_state.analysis_ran = False

    if st.button("Run analysis", type="primary"):
        st.session_state.analysis_ran = True

    if not st.session_state.analysis_ran:
        return

    if outcome.mode == "categorical":
        run_categorical_flow(
            df=df,
            uploaded_name=uploaded_name,
            roles=roles,
            pair=pair,
            y_value=outcome.y_value,
            include_decomposition=include_decomposition,
            variable_notes=variable_notes,
            llm=llm,
        )
    else:
        run_threshold_flow(
            df=df,
            uploaded_name=uploaded_name,
            roles=roles,
            pair=pair,
            outcome=outcome,
            include_decomposition=include_decomposition,
            variable_notes=variable_notes,
            llm=llm,
        )

    render_llm_output_from_state()


if __name__ == "__main__":
    main()
