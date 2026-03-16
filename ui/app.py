import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from functions import *
from openai import OpenAI

load_dotenv(override=True)  # loads OPENAI_API_KEY from .env

if os.getenv("OPENAI_API_KEY") is None:
    raise RuntimeError("OPENAI_API_KEY not found. ")

client = OpenAI()

def get_effective_orders(w_order, z_order, unknown_w_order: bool, unknown_z_order: bool):
    eff_w = None if unknown_w_order else (w_order or None)
    eff_z = None if unknown_z_order else (z_order or None)
    return eff_w, eff_z

def df_fingerprint(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df, index=True).sum())

def build_analysis_snapshot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x0_value,
    x1_values,
    y_mode: str,
    y_value=None,
    y_threshold=None,
) -> pd.DataFrame:
    out = df.copy()

    # restrict to the intended x: x0_value, x1_values
    keep = out[x_col].eq(x0_value) | out[x_col].isin(x1_values)
    out = out.loc[keep].copy()
    # binarize: 1 for x1_values, 0 for x0_value only
    out["__Xbin__"] = out[x_col].isin(x1_values).astype("int8")
    #  binarize Y
    if y_mode == "binary":
        out["__Ybin__"] = (out[y_col] == y_value).astype("int8")
    elif y_mode == "continuous":
        if y_threshold is None:
            raise ValueError("y_threshold must be provided for continuous mode snapshot.")
        out["__Ybin__"] = (out[y_col].to_numpy() <= float(y_threshold)).astype("int8")
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    return out
def compute_pairwise_effects_for_x_cond(
    df_with_ybin: pd.DataFrame,
    x_col: str,
    x0_value,
    x_cond,
    ybin_col: str,
    w_cols,
    z_cols,
    w_order=None,
    z_order=None,
):

    df_pair = df_with_ybin[df_with_ybin[x_col].isin([x0_value, x_cond])].copy()
    if df_pair.shape[0] == 0:
        raise ValueError("No rows for the selected pair (x0, x_cond).")

    df_pair["__Xbin_pair__"] = (df_pair[x_col] == x_cond).astype("int8")

    eff = compute_effects_multi(
        df=df_pair,
        x0=0, x1=1, y=1,
        x_col="__Xbin_pair__",
        y_col=ybin_col,
        w_cols=w_cols,
        z_cols=z_cols,
        w_order=w_order,
        z_order=z_order,
        do_decomposition=False
    )
    return eff



@st.cache_data(show_spinner=False)
def cached_compute_effects_multi(df_fp: str, df: pd.DataFrame, **kwargs):
    return compute_effects_multi(df=df, **kwargs)

@st.cache_data(show_spinner=False)
def detailed_effects_at_threshold(
    df_fp: str,
    df: pd.DataFrame,
    thr: float,
    x_col: str,
    y_col: str,
    x0_value,
    x1_values,
    w_cols,
    z_cols,
    unknown_w_order: bool,
    unknown_z_order: bool,
    w_order=None,
    z_order=None,
):
    keep = df[x_col].eq(x0_value) | df[x_col].isin(x1_values)
    df = df.loc[keep].copy()

    xbin = df[x_col].isin(x1_values).to_numpy().astype("int8")
    ybin = (df[y_col].to_numpy() <= thr).astype("int8")
    df_cs = df.assign(__Xbin__=xbin, __Ybin__=ybin)

    eff_w_order, eff_z_order = get_effective_orders(
        w_order=w_order,
        z_order=z_order,
        unknown_w_order=unknown_w_order,
        unknown_z_order=unknown_z_order,
    )


    eff = compute_effects_multi(
        df=df_cs,
        x0=0, x1=1, y=1,
        x_col="__Xbin__",
        y_col="__Ybin__",
        w_cols=w_cols,
        z_cols=z_cols,
        w_order=eff_w_order,
        z_order=eff_z_order,
        do_decomposition=True,
    )
    return eff.to_dict() if hasattr(eff, "to_dict") else eff


def compute_stepwise_effects(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols,
    y_value,
    x_order,
    unknown_w_order: bool,
    unknown_z_order: bool,
    w_order=None,
    z_order=None,
):
    if not x_order or len(x_order) < 2:
        return []

    eff_w_order, eff_z_order = get_effective_orders(
        w_order=w_order,
        z_order=z_order,
        unknown_w_order=unknown_w_order,
        unknown_z_order=unknown_z_order,
    )

    steps = []
    for i in range(len(x_order) - 1):
        x_from = x_order[i]
        x_to = x_order[i + 1]

        df_step = df[df[x_col].isin([x_from, x_to])].copy()

        x_bin = "__Xbin_step__"
        y_bin = "__Ybin_step__"
        df_step[x_bin] = (df_step[x_col] == x_to).astype(int)
        df_step[y_bin] = (df_step[y_col] == y_value).astype(int)

        eff = compute_effects_multi(
            df=df_step,
            x0=0, x1=1, y=1,
            x_col=x_bin,
            y_col=y_bin,
            w_cols=w_cols,
            z_cols=z_cols,
            w_order=eff_w_order,
            z_order=eff_z_order,
            do_decomposition=False
        )

        steps.append({
            "from": x_from,
            "to": x_to,
            "n_rows": int(df_step.shape[0]),
            "tv": eff.get("tv"),
            "te": eff.get("te"),
            "de": eff.get("de"),
            "ie": eff.get("ie"),
            "se": eff.get("se"),
        })
    return steps




def mark_dirty():
    st.session_state["analysis_dirty"] = True
    st.session_state["results"] = None  # discard stale results
    st.session_state["llm_text"] = None
    st.session_state["llm_latex"] = None

def init_state_defaults():
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "analysis_dirty" not in st.session_state:
        st.session_state["analysis_dirty"] = False
    if "llm_text" not in st.session_state:
        st.session_state["llm_text"] = None
    if "llm_latex" not in st.session_state:
        st.session_state["llm_latex"] = None
    

# Function analysis for both continouses and discrete case
def run_analysis(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols,
    x0_value,
    x1_values,
    y_value,
    user_explanation,
    unknown_w_order: bool,
    unknown_z_order: bool,
    w_order=None,
    z_order=None,
    use_stepwise=False,
    x_order=None,
    y_mode="binary",
    y_thresholds=None,
    progress_cb=None,
):

    df_fp = df_fingerprint(df)
    eff_w_order, eff_z_order = get_effective_orders(
        w_order=w_order,
        z_order=z_order,
        unknown_w_order=unknown_w_order,
        unknown_z_order=unknown_z_order,
    )
    # CONTINUOUS Y 
    if y_mode == "continuous":
        if y_thresholds is None or len(y_thresholds) == 0:
            raise ValueError("y_thresholds must be provided for continuous Y.")
        t0 = time.perf_counter()
        curves = compute_effects_continuous_y(
            df=df,
            x_col=x_col,
            y_col=y_col,
            w_cols=w_cols,
            z_cols=z_cols,
            x0_value=x0_value,
            x1_values=x1_values,
            y_thresholds=y_thresholds,
            w_order=eff_w_order,
            z_order=eff_z_order,
            progress_cb=progress_cb
        )
        print("compute_effects_continuous_y:", time.perf_counter() - t0)

        stepwise_curve = []

        return {
            "x_col": x_col,
            "y_col": y_col,
            "w_cols": w_cols,
            "z_cols": z_cols,
            "x0_value": x0_value,
            "x1_values": x1_values,
            "y_mode": "continuous",
            "y_thresholds": list(map(float, y_thresholds)),
            "effects_curve": curves,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "user_explanation": user_explanation,
            "stepwise": {
                "enabled": bool(use_stepwise),
                "x_order": list(x_order) if x_order is not None else None,
            },
        }

    y_bin_col = "__Ybin__"
    x_bin_col = "__Xbin__"

    keep = df[x_col].eq(x0_value) | df[x_col].isin(x1_values)
    df_cs = df.loc[keep].copy()

    df_cs = df_cs.assign(
        **{
            y_bin_col: (df_cs[y_col] == y_value).astype("int8"),
            x_bin_col: df_cs[x_col].isin(x1_values).astype("int8"),
        }
    )

    effects = cached_compute_effects_multi(
        df_fp,
        df_cs,
        x0=0,
        x1=1,
        y=1,
        W_values=None,
        x_col=x_bin_col,
        y_col=y_bin_col,
        w_cols=w_cols,
        z_cols=z_cols,
        do_decomposition=True,
        w_order=eff_w_order,
        z_order=eff_z_order,
       )


    if hasattr(effects, "to_dict"):
        effects_dict = effects.to_dict()
    elif isinstance(effects, dict):
        effects_dict = effects
    else:
        effects_dict = {"raw_effects": str(effects)}

    stepwise_effects = []
    if use_stepwise and x_order is not None:
          stepwise_effects = compute_stepwise_effects(
            df=df,
            x_col=x_col,
            y_col=y_col,
            w_cols=w_cols,
            z_cols=z_cols,
            y_value=y_value,
            x_order=x_order,
            unknown_w_order=unknown_w_order,
            unknown_z_order=unknown_z_order,
            w_order=w_order,
            z_order=z_order,
        )
        


    return {
        "x_col": x_col,
        "y_col": y_col,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "x0_value": x0_value,
        "x1_values": x1_values,
        "x1_value": x1_values[0] if (x1_values and len(x1_values) == 1) else None,
        "y_mode": "binary",
        "y_value": y_value,
        "user_explanation": user_explanation,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "stepwise": {
            "enabled": bool(use_stepwise),
            "x_order": list(x_order) if x_order is not None else None,
            "effects_by_step": stepwise_effects,
        },
        "effects": {
            "total_variation": effects_dict.get("tv"),
            "total_effect": effects_dict.get("te"),
            "indirect_effect": effects_dict.get("ie"),
            "direct_effect": effects_dict.get("de"),
            "spurious_effect": effects_dict.get("se"),
            "spurious effect decomposition": effects_dict.get("se_decomp_interval") or effects_dict.get("se_decomp") or {},
            "indirect effect decomposition": effects_dict.get("ie_decomp_interval") or effects_dict.get("ie_decomp") or {},

        },
        "effects_raw": effects_dict,
    }



def build_effect_tree(effects: dict) -> Digraph:
    dot = Digraph()
    dot.attr("node", shape="box", style="rounded,filled", fontsize="10")

    def fmt(label: str, key: str):
        val = effects.get(key)
        if val is None:
            return label
        return f"{label}\n({rounded_val(val, nd=5)})"

    # main structure
    dot.node("TV", fmt("TV", "total_variation"))
    dot.node("TE", fmt("TE", "total_effect"))
    dot.node("SE", fmt("SE", "spurious_effect"))
    dot.edge("TV", "TE")
    dot.edge("TV", "SE")

    dot.node("DE", fmt("DE", "direct_effect"))
    dot.node("IE", fmt("IE", "indirect_effect"))
    dot.edge("TE", "DE")
    dot.edge("TE", "IE")

    # IE decomposition:
    indirect_int = effects.get("ie_decomp_interval")
    if not indirect_int:
        indirect_int = effects.get("indirect effect decomposition")

    indirect_point = effects.get("ie_decomp")
    if not indirect_point:
        indirect_point = effects.get("indirect effect decomposition")

    indirect_decomp = indirect_int if (isinstance(indirect_int, dict) and len(indirect_int) > 0) else indirect_point

    if isinstance(indirect_decomp, dict) and len(indirect_decomp) > 0:
        for i, (name, val) in enumerate(indirect_decomp.items()):
            node_id = f"IE_{i}"
            dot.node(node_id, f"{name}\n({rounded_val(val, nd=5)})")
            dot.edge("IE", node_id)

    # SE decomposition
    spurious_int = effects.get("se_decomp_interval")
    if not spurious_int:
        spurious_int = effects.get("spurious effect decomposition")

    spurious_point = effects.get("se_decomp")
    if not spurious_point:
        spurious_point = effects.get("spurious effect decomposition")

    spurious_decomp = spurious_int if (isinstance(spurious_int, dict) and len(spurious_int) > 0) else spurious_point

    if isinstance(spurious_decomp, dict) and len(spurious_decomp) > 0:
        for j, (name, val) in enumerate(spurious_decomp.items()):
            node_id = f"SE_{j}"
            dot.node(node_id, f"{name}\n({rounded_val(val, nd=5)})")
            dot.edge("SE", node_id)

    return dot

# SFM GRAPH 
def build_sfm_graph(X, Y, W_vars, Z_vars):
    dot = Digraph()

    dot.attr(
        "graph",
        rankdir="TB",
        splines="true",
        center="true",
        nodesep="2",
        ranksep="1",
        size="10,5",
        scale="0.9",
    )
    dot.attr("node", fontsize="5")

    dot.node("X", X, shape="plaintext")
    dot.node("Y", Y, shape="plaintext")

    if Z_vars:
        z_lines = "   ".join(Z_vars)
        z_label = f"{z_lines}"
        dot.node("Z", z_label, shape="plaintext", fontsize="4", margin="0.6,0.6")

    if W_vars:
        w_lines = "   ".join(W_vars)
        w_label = f"{w_lines}"
        # dot.node("W", w_label, shape="box", fontsize="4")
        dot.node("W", w_lines, shape="plaintext", fontsize="4")

    if Z_vars:
        with dot.subgraph(name="rank_top") as s:
            s.attr(rank="min")
            s.node("Z")

    with dot.subgraph(name="rank_middle") as s:
        s.attr(rank="same")
        s.node("X")
        s.node("Y")

    if W_vars:
        with dot.subgraph(name="rank_bottom") as s:
            s.attr(rank="max")
            s.node("W")

    dot.edge("X", "Y")

    if W_vars:
        dot.edge("X", "W")
        dot.edge("W", "Y")

    if Z_vars:

        dot.edge("Z", "Y")
        dot.edge(
            "X",
            "Z",
            dir="both",
            arrowhead="normal",
            arrowtail="normal",
            style="dashed",
            headport="w",
            tailport="n",
            minlen="0.3",
        )
    if set(Z_vars) & set(W_vars):
        dot.edge("Z", "W")

    return dot


def _sync_state_with_options(key, options, multi=False):
    if multi:
        st.session_state[key] = [
            v for v in st.session_state.get(key, []) if v in options
        ]
    else:
        if st.session_state.get(key) not in options:
            st.session_state[key] = None
            
st.session_state.setdefault("x_col", None)
st.session_state.setdefault("y_col", None)
st.session_state.setdefault("w_cols", [])   
st.session_state.setdefault("z_cols", [])
st.session_state.setdefault("w_order", [])
st.session_state.setdefault("z_order", [])
st.session_state.setdefault("unknown_w_order", False)
st.session_state.setdefault("unknown_z_order", False)


def on_x_change():
    x = st.session_state.x_col
    if x:
        if st.session_state.y_col == x:
            st.session_state.y_col = None
        st.session_state.w_cols = [c for c in st.session_state.w_cols if c != x]
        st.session_state.z_cols = [c for c in st.session_state.z_cols if c != x]
    mark_dirty()


def on_y_change():
    y = st.session_state.y_col
    if y:
        if st.session_state.x_col == y:
            st.session_state.x_col = None
        st.session_state.w_cols = [c for c in st.session_state.w_cols if c != y]
        st.session_state.z_cols = [c for c in st.session_state.z_cols if c != y]
    mark_dirty()


def on_w_change():
    ws = set(st.session_state.w_cols or [])
    if st.session_state.x_col in ws:
        st.session_state.x_col = None
    if st.session_state.y_col in ws:
        st.session_state.y_col = None
    st.session_state.z_cols = [c for c in st.session_state.z_cols if c not in ws]
    mark_dirty()


def on_z_change():
    zs = set(st.session_state.z_cols or [])
    if st.session_state.x_col in zs:
        st.session_state.x_col = None
    if st.session_state.y_col in zs:
        st.session_state.y_col = None
    st.session_state.w_cols = [c for c in st.session_state.w_cols if c not in zs]
    mark_dirty()

def on_w_order_change():
    mark_dirty()

def on_unknown_w_order_change():
    # if user says they don't know, clear any existing order
    if st.session_state.get("unknown_w_order", False):
        st.session_state["w_order"] = []
    mark_dirty()

def on_z_order_change():
    mark_dirty()

def on_unknown_z_order_change():
    if st.session_state.get("unknown_z_order", False):
        st.session_state["z_order"] = []
    mark_dirty()


def summarize_with_llm_combined(results, client):
    results_json = json.dumps(results, indent=2)
    prompt_path = Path(__file__).parent / "fairmind_prompts.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    user_prompt = f"""
Here are the fairness decomposition results in JSON:

{results_json}
"""

    completion = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    full_output = completion.choices[0].message.content.strip()
    # Check if there's latex 
    if "LATEX:" not in full_output:
        raise ValueError("LLM output did not contain LATEX section.")

    text_part, latex_part = full_output.split("LATEX:", 1)
    text_part = text_part.replace("TEXT:", "").strip()
    latex_part = latex_part.strip()

    token_usage= completion.usage

    return text_part, latex_part, token_usage


# Main 

st.set_page_config(
    page_title="FairMind",
    layout="centered"
)

def main():
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

    init_state_defaults()
    if "results" not in st.session_state:
        st.session_state["results"] = None

    st.write(
        """
Upload a dataset for **causal fairness analysis**, specify:

- **X**: sensitive attribute (e.g. race, gender)
- **Y**: outcome (e.g. income)
- **W**: mediator(s) if any
- **Z**: confounder(s) if any

Assuming a specified SFM causal graph, the app will run a code to decompose the *total variation* into:
*total effect*, *indirect*, *direct*, and *spurious effects* and then the LLM will generate a **report** with the main results.
"""
    )

    # Uploading the file
    uploaded_file = st.file_uploader(
        "Upload your dataset", type=["csv", "tsv", "xlsx", "json"], key="uploaded_file"
    )

    if uploaded_file is None:
        st.info("Please upload a file to get started.")
        st.stop()

    data_status = st.radio(
        "What is the status of your dataset?",
        ("Processed", "Raw (removing NaN and invalid symbols)"),
        key="data_status",
    )

    try:
        file_suffix = Path(uploaded_file.name).suffix.lower()

        if file_suffix == ".csv":
            df = pd.read_csv(uploaded_file)

        elif file_suffix == ".tsv":
            df = pd.read_csv(uploaded_file, sep="\t")

        elif file_suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)

        elif file_suffix == ".json":
            df = pd.read_json(uploaded_file)

        else:
            st.error(f"Unsupported file type: {file_suffix}")
            return

    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

   

    # If raw, clean it 
    if data_status == "Raw (removing NaN and invalid symbols)":

        # Replace symbols with NaN
        # df.replace("?", np.nan, inplace=True)
        df.replace(r"^[^A-Za-z0-9]+$", np.nan, regex=True, inplace=True)

        # Drop rows with NaN
        old_rows = len(df)
        df.dropna(inplace=True)
        new_rows = len(df)

        st.success(
            f"Cleaning complete! Removed {old_rows - new_rows} rows "
            f"with missing values and invalid symbols."
        )

    else:
        st.info("Dataset marked as already processed. No cleaning applied.")
    w_cols_used = []
    z_cols_used = []
  

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    columns = list(df.columns)
    # To change the dataset
    _sync_state_with_options("x_col", columns)
    _sync_state_with_options("y_col", columns)
    _sync_state_with_options("w_cols", columns, multi=True)
    _sync_state_with_options("z_cols", columns, multi=True)

    st.subheader("Variable roles")
    # Initialize session state keys once, before using them
    for k, v in {
        "x_col": None,
        "y_col": None,
        "w_cols": [],
        "z_cols": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "X: sensitive attribute (e.g., race)",
            options=[None] + columns,
            index=(
                ([None] + columns).index(st.session_state.x_col)
                if st.session_state.x_col in columns
                else 0
            ),
            key="x_col",
            on_change=on_x_change,
            help="Chosen X will be removed from Y, W, Z automatically.",
        )

        st.selectbox(
            "Y: outcome (e.g., income > 50K)",
            options=[None] + columns,
            index=(
                ([None] + columns).index(st.session_state.y_col)
                if st.session_state.y_col in columns
                else 0
            ),
            key="y_col",
            on_change=on_y_change,
            help="Chosen Y will be removed from X, W, Z automatically.",
        )

    with col2:
        print(columns)
        print(st.session_state.w_cols)
        st.multiselect(
            "W: mediators",
            options=columns,
            key="w_cols",
            on_change=on_w_change,
            help="Any selection in W will be removed from X, Y, and Z.",
        )

        st.multiselect(
            "Z: confounders",
            options=columns,
            key="z_cols",
            on_change=on_z_change,
            help="Any selection in Z will be removed from X, Y, and W.",
        )

    used = (
        {c for c in [st.session_state.x_col, st.session_state.y_col] if c}
        | set(st.session_state.w_cols)
        | set(st.session_state.z_cols)
    )
    st.caption(
        f"Selected variables are kept unique across X, Y, W, Z. Currently used: {sorted(used)}"
    )

    x_col = st.session_state.get("x_col", None)
    y_col = st.session_state.get("y_col", None)
    w_cols = list(st.session_state.get("w_cols", []) or [])
    z_cols = list(st.session_state.get("z_cols", []) or [])


    x_col_used = x_col
    y_col_used = y_col
    w_cols_used = w_cols
    z_cols_used = z_cols
    # SFM graph
    show_sfm = st.toggle(
    "Show causal graph (SFM)",
    value=False,)

    if show_sfm:
        if x_col is None or y_col is None:
            st.error("Please select at least X and Y.")
        else:
            dot = build_sfm_graph(x_col, y_col, w_cols, z_cols)
            st.graphviz_chart(dot, use_container_width=False, width=400, height=300)
            st.write(
    "This is the causal graph you are assuming and passing to the identification step."
)
    
    
    
    st.subheader("**Stepwise effects**")
    
    st.caption("Stepwise decomposition breaks the total disparity into effects between adjacent "
        "levels of X.")
    st.info(
        "Enable stepwise effects only when the protected attribute X has a meaningful order "
        "(e.g. education level, income brackets, age groups). "
    )

    use_stepwise = st.toggle(
        "Enable stepwise effects",
        value=False,
        key="use_stepwise",
        on_change=mark_dirty,
        help="Use this only if X has a meaningful order (e.g., education level).",
    )

    x_order = None

    if use_stepwise:
        if x_col_used is None or x_col_used not in df.columns:
            st.warning("Select the protected attribute X first to enable stepwise effects.")
        else:
            x_unique_list = df[x_col_used].dropna().unique().tolist()

            is_numeric_x = pd.api.types.is_numeric_dtype(df[x_col_used])
            default_order = sorted(x_unique_list) if is_numeric_x else x_unique_list

            st.warning(
                "X is numeric; using ascending order by default."
                if is_numeric_x
                else "X is categorical. Please confirm the intended order."
            )

            x_order = st.multiselect(
                "Order X from lowest to highest (select ALL values)",
                options=x_unique_list,
                default=default_order,
                key="x_order",
                on_change=mark_dirty,
            )

            if set(x_order) != set(x_unique_list):
                st.error("Please select ALL categories.")
                x_order = None
            else:
                st.caption(f"Using order: {x_order}")

    # Grouped version 

    st.subheader("Choose order for W and Z decompositions")
    st.caption(
    "Specify an order **only if more than one mediator (W) or confounder (Z) is selected**. "
    
) 
    st.info("With a single variable, the order is fixed and no input is required.")

    # Topological order of the mediators
    if w_cols_used and len(w_cols_used) > 1:
        st.markdown("**Order of W (mediators) for IE decomposition**")
        st.markdown("### W Order")
        st.session_state.setdefault("unknown_w_order", False)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.checkbox("I don’t know", key="unknown_w_order", on_change=on_unknown_w_order_change)

        with col2:
            if st.session_state.unknown_w_order:
                st.caption("Order unknown → IE intervals will be shown.")
                st.session_state.w_order = []
            else:
                st.multiselect(
                    "Mediator order",
                    options=st.session_state.w_cols,
                    key="w_order",
                    on_change=on_w_order_change,
                )
        
    # Order for the Z same as W
    if z_cols_used and len(z_cols_used) > 1:
        st.markdown("**Order of Z (confounders) for SE decomposition**")
        st.session_state.setdefault("unknown_z_order", False)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.checkbox("I don’t know", key="unknown_z_order", on_change=on_unknown_z_order_change)

        with col2:
            if st.session_state.unknown_z_order:
                st.caption("Order unknown → SE intervals will be shown.")
                st.session_state.z_order = []
            else:
                st.multiselect(
                    "Confounder order",
                    options=st.session_state.z_cols,
                    key="z_order",
                    on_change=on_z_order_change,
                )
                
                


    # The user chooses which values are x0, x1, and y (I need them for identification formulas)
    st.subheader("Protected Attribute and Outcome Setup")
    x0_value = None
    x1_value = None
    x1_values = None  

    if x_col_used is not None:
        x_unique = sorted(df[x_col_used].dropna().unique().tolist())
        st.markdown(f"Unique values in **X ({x_col_used})**: `{x_unique}`")
        if len(x_unique) > 0:
            x0_value = st.selectbox(
                "x0 (protected / disadvantaged group)",
                options=x_unique,
                index=0,
                key="x0_value",
                on_change=mark_dirty,
            )
            
            x1_options = ["All remaining groups"] + [v for v in x_unique if v != x0_value]

            x1_choice = st.selectbox(
                "x1 (reference / advantaged group)",
                options=x1_options,
                index=0,
                key="x1_choice",
                on_change=mark_dirty,
            )

            if x1_choice == "All remaining groups":
                x1_values = [v for v in x_unique if v != x0_value]
                x1_value = None
                st.caption(f"x1 = all groups except `{x0_value}`: {x1_values}")
            else:
                x1_value = x1_choice
                x1_values = [x1_value]


                    

    # Outcome type: Continous or Discrete/ Categorical
    y_mode = st.radio(
    "**Outcome Y type** ",
    options=["Binary/Categorical (choose favourable value)", "Continuous (threshold y)"],
    key="y_mode",
    on_change=mark_dirty,
)
    y_value = None
    y_thresholds = None
    if y_col_used is not None:
        sY = df[y_col_used].dropna()

        if y_mode.startswith("Binary"):
            y_unique = sorted(sY.unique().tolist())
            st.markdown(f"Unique values in **Y ({y_col_used})**: `{y_unique}`")
            y_value = st.selectbox(
                "y (favourable outcome value, e.g. income > 50K = 1)",
                options=y_unique,
                index=min(1, len(y_unique) - 1) if len(y_unique) > 1 else 0,
                key="y_value",
                on_change=mark_dirty,
            )
        else:
            if not pd.api.types.is_numeric_dtype(sY):
                st.error("Continuous mode requires Y to be numeric.")
            else:
                st.markdown(f"**Y ({y_col_used})** numeric range: [{sY.min()}, {sY.max()}]")

                grid_kind = st.selectbox(
                    "Threshold grid",
                    ["Quantiles", "Evenly spaced"],
                    key="y_grid_kind",
                    on_change=mark_dirty,
                )
                n_points = st.slider(
                    "Number of thresholds",
                    min_value=5, max_value=200, value=25,
                    key="y_n_points",
                    on_change=mark_dirty,
                )

                if grid_kind == "Quantiles":
                    qs = np.linspace(0.01, 0.99, n_points)
                    y_thresholds = np.quantile(sY.to_numpy(), qs)
                else:
                    y_thresholds = np.linspace(float(sY.min()), float(sY.max()), n_points)
                y_thresholds = np.unique(y_thresholds)        # remove duplicates (many 0s!)
                y_thresholds = y_thresholds.astype(float)      

                st.caption(f"Using {len(y_thresholds)} thresholds.")
   
    user_explanation = st.text_area(
        "Explain the meaning of X, Y, W, Z (recommended if the variables are not self-explainatory)",
        placeholder=(
            "For example: X is race (0 = non-White, 1 = White), "
            "Y is income (1 = income > 50K), "
            "W mediators are education and experience, Z is a confounder..."
        ),
        key="user_explanation",
        on_change=mark_dirty,
    )

    if st.button("Run causal fairness analysis"):
        if x_col_used is None or y_col_used is None:
            st.error("Please select at least X and Y.")
            return
        if x0_value is None or (not x1_values):
            st.error("Please specify x0 and x1 (or use all-others).")
            return

        if y_mode.startswith("Binary"):
            if y_value is None:
                st.error("Please specify the favourable outcome value y.")
                return
        else:  # continuous Y
            if y_thresholds is None or len(y_thresholds) == 0:
                st.error("Please specify at least one threshold for continuous Y.")
                return
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.perf_counter()
        def progress_cb(done, total, thr):
            frac = done / total if total else 1.0
            progress_bar.progress(frac)
            elapsed = time.perf_counter() - start_time
            rate = elapsed / done if done else None          # seconds per threshold
            eta = rate * (total - done) if rate is not None and total else None
            def fmt_s(s):
                if s is None:
                    return "—"
                s = int(round(s))
                m, sec = divmod(s, 60)
                h, m = divmod(m, 60)
                return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
           # status_text.write(f"Threshold {done}/{total} — y <= {thr:.6g}")
            status_text.write(f"Threshold {done}/{total} — binarization at y ≤ {thr:.6g} " f"• elapsed {fmt_s(elapsed)} • ETA {fmt_s(eta)}")


        with st.spinner("Computing causal fairness decomposition..."):
            try:
                t0 = time.perf_counter()                 
                unknown_w = bool(st.session_state.get("unknown_w_order", False))
                unknown_z = bool(st.session_state.get("unknown_z_order", False))
                w_order_ss = list(st.session_state.get("w_order", []) or [])
                z_order_ss = list(st.session_state.get("z_order", []) or [])

                results = run_analysis(
                            df=df,
                            x_col=x_col_used,
                            y_col=y_col_used,
                            w_cols=w_cols_used,
                            z_cols=z_cols_used,
                            x0_value=x0_value,
                            x1_values=x1_values,
                            y_value=y_value,
                            y_mode="continuous" if y_mode.startswith("Continuous") else "binary",
                            y_thresholds=y_thresholds,
                            use_stepwise=use_stepwise,
                            x_order=x_order,
                            user_explanation=user_explanation,
                            progress_cb=progress_cb,
                            unknown_w_order=unknown_w,
                            unknown_z_order=unknown_z,
                            w_order=w_order_ss,
                            z_order=z_order_ss,
                        )
                st.caption(f"run_analysis took {time.perf_counter() - t0:.2f}s")
                
            except Exception as e:
                st.error(f"Error in compute_effects_multi / run_analysis: {e}")
                return
        progress_bar.progress(1.0)
        status_text.write("Done.")
        st.session_state["results"] = results
        st.session_state["analysis_dirty"] = False
        st.success("Analysis completed. See the results below.")

    if st.session_state["results"] is not None:
        if st.session_state["analysis_dirty"]:
            st.warning(
                "Inputs have changed since the last run. Please press **Run causal fairness analysis** again."
            )
            return
        
        results = st.session_state["results"]
        stepwise = results.get("stepwise", {})
        # Y continuous
        if results.get("y_mode") == "continuous":
            st.subheader("Continuous-Y decomposition")
            st.markdown("**Decomposition curves over thresholds**")
            # Curve + table
            df_curve_full = pd.DataFrame(results["effects_curve"])
            df_curve_view = df_curve_full[["y_threshold", "tv", "te", "de", "ie", "se"]].copy()
            df_curve_view["y_threshold"] = df_curve_view["y_threshold"].round(6)
            st.dataframe(df_curve_view, use_container_width=True)

            st.line_chart(
                df_curve_view.set_index("y_threshold")[["tv", "te", "de", "ie", "se"]]
            )

            # Pick one threshold 
            thr_values = df_curve_full["y_threshold"].tolist()
            thr_selected_main = st.select_slider(
                "**Pick a threshold to inspect detailed effects (used for table/tree, x-specific, z-specific, LLM report)**",
                options=thr_values,
                value=thr_values[len(thr_values) // 2],
                key="thr_selected_main",
            )
            
            snap = df_curve_full.loc[
            df_curve_full["y_threshold"] == thr_selected_main].iloc[0].to_dict()

            with st.spinner("Computing detailed effects at selected threshold..."):
                t1 = time.perf_counter()
                unknown_w = bool(st.session_state.get("unknown_w_order", False))
                unknown_z = bool(st.session_state.get("unknown_z_order", False))
                w_order_ss = list(st.session_state.get("w_order", []) or [])
                z_order_ss = list(st.session_state.get("z_order", []) or [])

                eff_detail = detailed_effects_at_threshold(
                    df_fp=df_fingerprint(df),
                    df=df,
                    thr=float(thr_selected_main),
                    x_col=x_col_used,
                    y_col=y_col_used,
                    x1_values=x1_values,
                    x0_value=x0_value,     
                    w_cols=w_cols_used,
                    z_cols=z_cols_used,
                    unknown_w_order=unknown_w,
                    unknown_z_order=unknown_z,
                    w_order=w_order_ss,
                    z_order=z_order_ss,
                )
                st.session_state["eff_detail_selected_thr"] = eff_detail

                elapsed = time.perf_counter() - t1

            st.caption(f"Detailed threshold computation took {elapsed:.2f}s")

            effects = {
            "total_variation": snap.get("tv"),      
            "total_effect": eff_detail.get("te"),
            "direct_effect": eff_detail.get("de"),
            "indirect_effect": eff_detail.get("ie"),
            "spurious_effect": eff_detail.get("se"),
            "indirect effect decomposition": eff_detail.get("ie_decomp_interval") or eff_detail.get("ie_decomp") or {},
            "spurious effect decomposition": eff_detail.get("se_decomp_interval") or eff_detail.get("se_decomp") or {},
        }
            st.info(f"Detailed view at threshold y = {thr_selected_main}")

       
            df_cont_snapshot = build_analysis_snapshot(
                                    df=df,
                                    x_col=x_col_used,
                                    y_col=y_col_used,
                                    x0_value=x0_value,
                                    x1_values=x1_values,
                                    y_mode="continuous",
                                    y_threshold=float(thr_selected_main),
                                )

            st.session_state["df_cont_snapshot"] = df_cont_snapshot
            st.session_state["cont_thr_selected"] = float(thr_selected_main)

            st.subheader("Decomposition Table (selected threshold)")

            rows = []

            main_keys = [
                "total_variation",
                "total_effect",
                "indirect_effect",
                "direct_effect",
                "spurious_effect",
            ]

            for k in main_keys:
                if k in effects:
                    rows.append({"Component": k, "Value": rounded_val(effects[k], nd=5)})

            # Spurious decomposition
            spurious = effects.get("spurious effect decomposition", {})
            if isinstance(spurious, dict) and len(spurious) > 0:
                for name, val in spurious.items():
                    rows.append({"Component": f"spurious: {name}", "Value": rounded_val(val, nd=5)})
                

            # Indirect decomposition
            indirect = effects.get("indirect effect decomposition", {})
            if isinstance(indirect, dict) and len(indirect) > 0:
                for name, val in indirect.items():
                    rows.append({"Component": f"indirect: {name}", "Value": rounded_val(val, nd=5)})
            x1_label = ", ".join(map(str, x1_values))
            st.markdown(
                f"""
                <span>Effects computed between </span>
                <span style="color:#2e7d32; font-weight:600;">{x_col_used}</span>
                <span>: </span>
                <span style="color:#1565c0; font-weight:600;">[{x0_value}]</span>
                <span style="color:#444;"> vs </span>
                <span style="color:#1565c0; font-weight:600;">[{x1_label}]</span>
                """,
                unsafe_allow_html=True,
            )

            #st.markdown(f"Effects computed between: {contrast}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            

        

        if results.get("y_mode")!="continuous":
            effects=results["effects"]
        
        # TABLE
        if results.get("y_mode") != "continuous":
            st.subheader("Decomposition Table")
            rows = []

            main_keys = [
            "total_variation",
            "total_effect",
            "indirect_effect",
            "direct_effect",
            "spurious_effect",

        ]
            for k in main_keys:
                if k in effects:
                    rows.append({"Component": k, "Value": rounded_val(effects[k], nd=5)})

            # with spurios decomp (if not empty)
            spurious = effects.get("spurious effect decomposition", {})
            if isinstance(spurious, dict) and len(spurious) > 0:
                for name, val in spurious.items():
                    rows.append({"Component": f"spurious: {name}", "Value": rounded_val(val, nd=5)})
                

            # with indirect decomp (if not empty)
            indirect = effects.get("indirect effect decomposition", {})
            if isinstance(indirect, dict) and len(indirect) > 0:
                for name, val in indirect.items():
                    rows.append({"Component": f"indirect: {name}", "Value": rounded_val(val, nd=5)})


            df_effects = pd.DataFrame(rows)
            x1_label = ", ".join(map(str, x1_values))
            st.markdown(
                f"""
                <span>Effects computed between </span>
                <span style="color:#2e7d32; font-weight:600;">{x_col_used}</span>
                <span>: </span>
                <span style="color:#1565c0; font-weight:600;">[{x0_value}]</span>
                <span style="color:#444;"> vs </span>
                <span style="color:#1565c0; font-weight:600;">[{x1_label}]</span>
                """,
                unsafe_allow_html=True,
            )


            st.dataframe(df_effects, use_container_width=True)
            

        # TREE
        st.subheader("Effect Decomposition Tree")
        effects_for_tree = dict(effects)  

        if results.get("y_mode") == "continuous":
            eff_detail_saved = st.session_state.get("eff_detail_selected_thr", None)
            if isinstance(eff_detail_saved, dict):
                effects_for_tree.update(eff_detail_saved)
        else:
            raw = results.get("effects_raw", {})
            if isinstance(raw, dict):
                effects_for_tree.update(raw)

        dot = build_effect_tree(effects_for_tree)
        st.graphviz_chart(dot)


        # X-specific and Z-specific effects
        st.subheader("Optional: Specific Effects")
        unknown_w = bool(st.session_state.get("unknown_w_order", False))
        unknown_z = bool(st.session_state.get("unknown_z_order", False))
        w_order_ss = list(st.session_state.get("w_order", []) or [])
        z_order_ss = list(st.session_state.get("z_order", []) or [])
        eff_w_order, eff_z_order = get_effective_orders(
            w_order=w_order_ss,
            z_order=z_order_ss,
            unknown_w_order=unknown_w,
            unknown_z_order=unknown_z,
        )

        if results.get("y_mode") == "continuous":
            df_spec = st.session_state.get("df_cont_snapshot", None)
            if df_spec is None or "__Ybin__" not in df_spec.columns:
                st.error("Missing continuous snapshot (__Ybin__). Please re-run analysis and select a threshold.")
                return
            ybin_col = "__Ybin__"
        else:
            df_spec = build_analysis_snapshot(
                df=df,
                x_col=x_col_used,
                y_col=y_col_used,
                x0_value=x0_value,
                x1_values=x1_values,
                y_mode="binary",
                y_value=y_value,
            )
            ybin_col = "__Ybin__"

        tabs = ["X-specific effects"]
        if z_cols_used:
            tabs.append("Z-specific effects")

        tab_objs = st.tabs(tabs)

        with tab_objs[0]:
            x_unique_used = sorted(df[x_col_used].dropna().unique().tolist())

            ALL_REMAINING = "All remaining groups"
            x_options = [ALL_REMAINING] + x_unique_used

            # Defaults: suggest x0 and (if single x1 was chosen) that x1, else just x0
            default_x = [x0_value] if x0_value in x_unique_used else ([x_unique_used[0]] if x_unique_used else [])
            if (x1_value is not None) and (x1_value in x_unique_used) and (x1_value not in default_x):
                default_x = default_x + [x1_value]

            selected_x_vals_raw = st.multiselect(
                f"Select X values to compare against x0 = {x0_value} (column: {x_col_used})",
                options=x_options,
                default=default_x,
                key="x_specific_selected_vals",
            )

            # Expand "All remaining groups" -> all values except x0
            if ALL_REMAINING in selected_x_vals_raw:
                selected_x_vals = [v for v in x_unique_used if v != x0_value]
                st.caption(f"{ALL_REMAINING} = all groups except `{x0_value}`: {selected_x_vals}")
            else:
                selected_x_vals = [v for v in selected_x_vals_raw if v != ALL_REMAINING]

            # Remove x0 itself if user selected it
            selected_x_vals = [v for v in selected_x_vals if v != x0_value]

            if not selected_x_vals:
                st.info("Select one or more X values (other than x0) to compute X-specific effects.")
            else:
                x_rows = []
                for x_cond in selected_x_vals:
                    try:
                        eff = compute_pairwise_effects_for_x_cond(
                            df_with_ybin=df_spec,
                            x_col=x_col_used,
                            x0_value=x0_value,
                            x_cond=x_cond,
                            ybin_col=ybin_col,
                            w_cols=w_cols_used,
                            z_cols=z_cols_used,
                            w_order=eff_w_order,
                            z_order=eff_z_order,
                        )
                        eff_dict = eff.to_dict() if hasattr(eff, "to_dict") else (eff if isinstance(eff, dict) else {})
                    except Exception as e:
                        st.error(f"Error computing X-specific effects for X={x_cond}: {e}")
                        continue

                    x_rows.append({
                        "x0": x0_value,
                        "x1": x_cond,
                        "n_rows": int(df_spec[df_spec[x_col_used].isin([x0_value, x_cond])].shape[0]),
                        "tv": eff_dict.get("tv"),
                        "te": eff_dict.get("te"),
                        "ie": eff_dict.get("ie"),
                        "de": eff_dict.get("de"),
                        "se": eff_dict.get("se"),
                    })

                st.session_state["x_specific_rows"] = x_rows
                st.markdown("**X-specific Effects (pairwise vs x0, with the same Y binarization as the main analysis)**")
                st.dataframe(pd.DataFrame(x_rows), use_container_width=True)

        if z_cols_used:
            with tab_objs[1]:
                st.markdown(
                    f"Z-specific effects computed using confounders: `{z_cols_used}`. "
                    "Effects are returned for all observed Z combinations."
                )

                try:
                    df_z = df_spec.copy()
                    if "__Xbin__" not in df_z.columns:
                        df_z["__Xbin__"] = df_z[x_col_used].isin(x1_values).astype("int8")

                    effects_z = compute_z_specific_effects(
                        df=df_z,
                        x0=0,
                        x1=1,
                        y_val=1,
                        x_col="__Xbin__",
                        y_col=ybin_col,
                        w_cols=w_cols_used,
                        z_cols=z_cols_used,
                    )
                except Exception as e:
                    st.error(f"Error computing Z-specific effects: {e}")
                else:
                    if not effects_z:
                        st.info("No Z-specific effects were returned.")
                    else:
                        rows_z = []
                        for z_tuple, eff_dict in effects_z.items():
                            row = {name: val for name, val in zip(z_cols_used, z_tuple)}
                            row.update(eff_dict)
                            rows_z.append(row)

                        st.session_state["z_specific_rows"] = rows_z

                        df_table = pd.DataFrame(rows_z)
                        preferred_order = list(z_cols_used) + ["tv", "te", "ie", "de", "se"]
                        ordered = [c for c in preferred_order if c in df_table.columns]
                        remaining = [c for c in df_table.columns if c not in ordered]
                        df_table = df_table[ordered + remaining]

                        st.markdown("**Z-specific Effects (consistent with main binarized X/Y)**")
                        st.dataframe(df_table, use_container_width=True)

        st.subheader("Stepwise effects")
        
        if results.get("stepwise", {}).get("enabled"):
            st.info(f"Stepwise enabled. X order: {results['stepwise']['x_order']}")
        else:
            st.warning("Stepwise effects not enabled.")

        if results.get("y_mode") == "continuous":
            steps = st.session_state.get("stepwise_effects_cont", []) or []
        else:
            steps = stepwise.get("effects_by_step") or []
        if stepwise.get("enabled") and results.get("y_mode") != "continuous":
            if not steps:
                st.info("Stepwise is enabled, but no stepwise results were computed.")
            else:
                def _sum(key):
                    return sum(s.get(key, 0.0) for s in steps if s.get(key) is not None)

                x_from = steps[0].get("from")
                x_to = steps[-1].get("to")
                for s in steps:
                    title = f"{s['from']} → {s['to']}  (n={s['n_rows']})"
                    with st.expander(title, expanded=False):
                        cols = st.columns(5)
                        def _fmt(v): return f"{v:.4f}" if v is not None else "—"
                        cols[0].markdown(f"**TV**<br>{_fmt(s.get('tv'))}", unsafe_allow_html=True)
                        cols[1].markdown(f"**TE**<br>{_fmt(s.get('te'))}", unsafe_allow_html=True)
                        cols[2].markdown(f"**DE**<br>{_fmt(s.get('de'))}", unsafe_allow_html=True)
                        cols[3].markdown(f"**IE**<br>{_fmt(s.get('ie'))}", unsafe_allow_html=True)
                        cols[4].markdown(f"**SE**<br>{_fmt(s.get('se'))}", unsafe_allow_html=True)

        # Stepwise (continuous) 
        #if use_stepwise and x_order is not None:
        if results.get("y_mode") == "continuous" and use_stepwise and x_order is not None:
            st.caption(f"Stepwise computed at y <= {float(thr_selected_main):.6g}")

            df_step_snapshot = df.assign(
                __Ybin__=(df[y_col_used].to_numpy() <= float(thr_selected_main)).astype("int8")
            )
            unknown_w = bool(st.session_state.get("unknown_w_order", False))
            unknown_z = bool(st.session_state.get("unknown_z_order", False))
            w_order_ss = list(st.session_state.get("w_order", []) or [])
            z_order_ss = list(st.session_state.get("z_order", []) or [])

            stepwise_effects = compute_stepwise_effects(
                df=df_step_snapshot,
                x_col=x_col_used,
                y_col="__Ybin__",
                w_cols=w_cols_used,
                z_cols=z_cols_used,
                y_value=1,
                x_order=x_order,
                unknown_w_order=unknown_w,
                unknown_z_order=unknown_z,
                w_order=w_order_ss,
                z_order=z_order_ss,
            )

            st.session_state["stepwise_effects_cont"] = stepwise_effects

            if not stepwise_effects:
                st.info("Stepwise enabled, but no stepwise results were computed.")
            else:
                # Optional cumulative summary
                def _sum(key):
                    vals = [s.get(key) for s in stepwise_effects if s.get(key) is not None]
                    return float(np.sum(vals)) if vals else None

                 #with st.expander("Cumulative stepwise effects", expanded=False):
                     #cols = st.columns(5)
                    #def _fmt(v): return f"{v:.4f}" if v is not None else "—"
                    #cols[0].markdown(f"**TV**<br>{_fmt(_sum('tv'))}", unsafe_allow_html=True)
                    #cols[1].markdown(f"**TE**<br>{_fmt(_sum('te'))}", unsafe_allow_html=True)
                    #cols[2].markdown(f"**DE**<br>{_fmt(_sum('de'))}", unsafe_allow_html=True)
                    #cols[3].markdown(f"**IE**<br>{_fmt(_sum('ie'))}", unsafe_allow_html=True)
                    #cols[4].markdown(f"**SE**<br>{_fmt(_sum('se'))}", unsafe_allow_html=True)

                # Per-step details
                for s in stepwise_effects:
                    title = f"{s['from']} → {s['to']}  (n={s['n_rows']})"
                    with st.expander(title, expanded=False):
                        cols = st.columns(5)
                        def _fmt(v): return f"{v:.4f}" if v is not None else "—"
                        cols[0].markdown(f"**TV**<br>{_fmt(s.get('tv'))}", unsafe_allow_html=True)
                        cols[1].markdown(f"**TE**<br>{_fmt(s.get('te'))}", unsafe_allow_html=True)
                        cols[2].markdown(f"**DE**<br>{_fmt(s.get('de'))}", unsafe_allow_html=True)
                        cols[3].markdown(f"**IE**<br>{_fmt(s.get('ie'))}", unsafe_allow_html=True)
                        cols[4].markdown(f"**SE**<br>{_fmt(s.get('se'))}", unsafe_allow_html=True)
    
        # LLM report 
        st.subheader("Generate textual fairness report")
        if results.get("y_mode") == "continuous":
            thr = st.session_state.get("cont_thr_selected")

            results_for_llm = {
                "analysis_type": "continuous",
                "selected_threshold": thr,
                "x_col": results["x_col"],
                "y_col": results["y_col"],
                "w_cols": results["w_cols"],
                "z_cols": results["z_cols"],
                "effects": effects,  
                "stepwise": results.get("stepwise"),
                "user_explanation": results.get("user_explanation"),
                "note": (
                    "Y was binarized as 1{Y <= selected_threshold} "
                    "for interpretation and fairness decomposition."
                ),
            }
        else:
            results_for_llm = results


        if st.button("Generate LLM report", key="generate_llm_report_2"):
            with st.spinner("Asking the LLM for a fairness-oriented explanation..."):
                try:
                    # Attach x/z specific (even if empty)
                    results_for_llm["x_specific"] = st.session_state.get("x_specific_rows", [])
                    results_for_llm["z_specific"] = st.session_state.get("z_specific_rows", [])

                    # Continuous: attach curve info
                    if results_for_llm.get("analysis_type") == "continuous":
                        dfc = pd.DataFrame(results["effects_curve"])[
                            ["y_threshold", "tv", "te", "de", "ie", "se"]
                        ]

                        mid = len(dfc) // 2
                        curve_sample = pd.concat(
                            [
                                dfc.head(5),
                                dfc.iloc[max(0, mid - 2): mid + 3],
                                dfc.tail(5),
                            ],
                            ignore_index=True,
                        )
                        results_for_llm["effects_curve_sample"] = curve_sample.to_dict(orient="records")

                        thr = float(st.session_state.get("cont_thr_selected"))
                        idx = (dfc["y_threshold"] - thr).abs().idxmin()
                        results_for_llm["selected_threshold_row"] = dfc.loc[idx].to_dict()

                    else:
                        # Binary/categorical: use the full results object
                        results_for_llm = results
                        results_for_llm["analysis_type"] = "binary"   
                        results_for_llm["x_specific"] = st.session_state.get("x_specific_rows", [])
                        results_for_llm["z_specific"] = st.session_state.get("z_specific_rows", [])


                    if results_for_llm.get("stepwise", {}).get("enabled"):
                        if results_for_llm.get("analysis_type") == "continuous":
                            results_for_llm["stepwise"]["effects_by_step"] = st.session_state.get(
                                "stepwise_effects_cont", []
                            )
                 

                    text, latex_doc,token_usage = summarize_with_llm_combined(results_for_llm, client)

                    st.session_state["llm_text"] = text
                    st.session_state["llm_latex"] = latex_doc
                    

                except Exception as e:
                    st.error(f"Error generating report: {e}")

        
        if st.session_state.get("llm_text"):
            st.markdown(st.session_state["llm_text"])

        if st.session_state.get("llm_latex"):
            with st.expander("Show LaTeX source"):
                st.code(st.session_state["llm_latex"], language="latex")

            st.download_button(
                "Download LaTeX (.tex)",
                st.session_state["llm_latex"].encode("utf-8"),
                file_name="fairness_report.tex",
                mime="application/x-tex",
            )
            st.markdown("## Token Usage Summary")
            st.markdown(f"Token Usage: {token_usage}")


if __name__ == "__main__":
    main()
