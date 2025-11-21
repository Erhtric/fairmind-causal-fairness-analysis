import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from functions import *

load_dotenv(override=True)  #loads OPENAI_API_KEY from .env

if os.getenv("OPENAI_API_KEY") is None:
    raise RuntimeError("OPENAI_API_KEY not found. ")

client = OpenAI()


def run_analysis(
    df,
    x_col,
    y_col,
    w_cols,
    z_cols,
    x0_value,
    x1_value,
    y_value,
    user_explanation,
):
    df_cs = df.copy()

    W_values = None

    effects = compute_effects_multi(
        df=df_cs,
        x0=x0_value,        
        x1=x1_value,        
        y=y_value,          
        W_values=W_values,
        x_col=x_col,
        y_col=y_col,
        w_cols=w_cols,
        z_cols=z_cols,
        do_decomposition=True,
    )

    if hasattr(effects, "to_dict"):
        effects_dict = effects.to_dict()
    elif isinstance(effects, dict):
        effects_dict = effects
    else:
        effects_dict = {"raw_effects": str(effects)}

   
    results = {
        "x_col": x_col,
        "y_col": y_col,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "x0_value": x0_value,
        "x1_value": x1_value,
        "y_value": y_value,
        "user_explanation": user_explanation,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "effects": {
            "total_variation": effects_dict.get("tv"),
            "total_effect": effects_dict.get("te"),
            "indirect_effect": effects_dict.get("ie"),
            "direct_effect": effects_dict.get("de"),
            "spurious_effect": effects_dict.get("se"),
            "spurious effect decomposition":effects_dict.get("se_decomp"),
            "indirect effect decomposition":effects_dict.get("ie_decomp"),
        },
        "effects_raw": effects_dict,  
    }
    return results


#llm
def summarize_with_llm(results):
    """
    Call the LLM to produce a human-readable summary of the fairness decomposition.
    """
    results_json = json.dumps(results, indent=2)

    system_prompt = (
        "You are a statistician and causal fairness expert. "
        "You receive a JSON object describing a causal fairness decomposition for a binary sensitive attribute X "
        "and a (usually binary) outcome Y. The analysis includes total variation, total effect, indirect effect, "
        "direct effect, and spurious effect, possibly with control variables W and instruments Z. "
        "Write a clear, concise explanation understandable by a data scientist, not a causal theory expert. "
        "Explain what X and Y represent (protected group vs reference group, favourable outcome), what each effect "
        "means in fairness terms (e.g. part of disparity explained by legitimate pathways vs potentially unfair paths), "
        "and how to interpret the signs and magnitudes. If we declared more than one mediator or confounded variables, explain the decomposition of the indirect and spurious effect accordingly."
    )

    user_prompt = f"""Here are the analysis results in JSON:

{results_json}

Write:

1. A short paragraph (3–6 sentences) describing:
   - what the fairness analysis is about,
   - which groups are compared (X=x0 vs X=x1),
   - what the outcome Y represents,
   - and what was decomposed.

2. A bullet list of important, focusing on:
   - total variation in outcome between the two groups,
   - total effect,
   - indirect effect (remember this is considering as baseline x1 while all the other considering baseline x0) please consider decomposing it if more than 1 mediator,
   - direct effect,
   - spurious effect (also decomposed if present),
   and their interpretation in terms of fairness and potential discrimination.


"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content.strip()


#user interface
def main():
    st.title("Causal Fairness Analysis")
    if "results" not in st.session_state:
        st.session_state["results"] = None

    st.write(
        """
Upload a dataset for **causal fairness analysis**, specify:

- **X**: sensitive attribute (e.g. race, gender)
- **Y**: outcome (e.g. income)
- **W**: mediator(s) if any
- **Z**: confounder(s) if any

The app will run a code to decompose the *total variation* into:
*total effect*, *indirect*, *direct*, and *spurious effects* and then the LLM will generate a **report** with the main results.
"""
    )

    #Uploading the file
    uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "tsv", "xlsx", "json"]  #not only CSV
)


    if uploaded_file is None:
        st.info("Please upload a file to get started.")
        return
    data_status = st.radio(
    "What is the status of your dataset?",
    ("Processed", "Raw (removing NaN and invalid symbols)")
)
    
    #read the data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

#if raw, clean it here
    if data_status == "Raw (removing NaN and invalid symbols)":
       

#Replace "?" or any other symbols except from numbers and letters with NaN
       # df.replace("?", np.nan, inplace=True)
        df.replace(r'^[^A-Za-z0-9]+$', np.nan, regex=True, inplace=True)


#Drop rows with at least one NaN
        old_rows = len(df)
        df.dropna(inplace=True)
        new_rows = len(df)
        st.success(f"Cleaning complete! Removed {old_rows - new_rows} rows with missing values and invalid symbols.")
    else:
        st.info("Dataset marked as already processed. No cleaning applied.")


    st.subheader("Dataset preview")
    st.dataframe(df.head())
 

    columns = list(df.columns)

    st.subheader("Variable roles")

#select X, Y, W, Z for the SFM
    col1, col2 = st.columns(2)

    with col1:
    #X: any column
        x_col = st.selectbox(
        "X: sensitive attribute (e.g., race)",
        options=[None] + columns,
    )

    #Y: different from X
        y_options = [c for c in columns if c != x_col]
        y_col = st.selectbox(
        "Y: outcome (e.g., income > 50K)",
        options=[None] + y_options,
    )

    with col2:
        #W, Z cannot reuse X or Y!!!!  They also need to be different from each other.
        base_wz_options = [c for c in columns if c not in {x_col, y_col}]

        w_cols = st.multiselect(
        "W: mediators",
        options=base_wz_options,
    )


        z_options = [c for c in base_wz_options if c not in w_cols]
        z_cols = st.multiselect(
        "Z: confounders",
        options=z_options,
    )



    #Numerical variable, maybe grouping
    st.subheader("Variable grouping (optional)")
    st.caption(
        "Numeric variables with more than 2 unique values can be grouped "
        "(e.g. create `age_group` from `age`). "
        "Binary or categorical variables are used as they are."
    )

    #collect selected variables
    all_vars = []
    for v in [x_col, y_col]:
        if v is not None:
            all_vars.append(v)
    for v in w_cols + z_cols:
        if v is not None:
            all_vars.append(v)

    #remove duplicates, keep order
    all_vars = list(dict.fromkeys(all_vars))

    grouping_map = {}  

    for var in all_vars:
        s = df[var].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(s)
        n_unique = s.nunique()


        #we're grouping only (categorical) variables
        if (not is_numeric) or (n_unique <= 2):
            grouping_map[var] = var
            continue

        #show this for numeric vars with > 2 unique values
        with st.expander(f"{var} – grouping and unique values", expanded=False):
            choice = st.radio(
                "How do you want to use this variable?",
                ("Use as is", "Bin into 3 equal-width groups", "Bin into 4 quantile groups"),
                key=f"group_choice_{var}",
            )

            if choice == "Use as is":
                used_col = var
            elif choice == "Bin into 3 equal-width groups":
                used_col = f"{var}_group"
                df[used_col] = pd.cut(s, bins=3)
            else:  #4 quantile groups
                used_col = f"{var}_group"
                df[used_col] = pd.qcut(s, q=4, duplicates="drop")

            grouping_map[var] = used_col

            # show unique values of the column that will actually be used
            uniques = df[used_col].dropna().unique().tolist()
            st.markdown(f"**Unique values in `{used_col}`**:")
            st.code(str(uniques), language="python")


    #grouped version (not needed for X, Y but ok) 
    # TODO: fix it, not necessary
    x_col_used = grouping_map.get(x_col, x_col)
    y_col_used = grouping_map.get(y_col, y_col)
    w_cols_used = [grouping_map.get(w, w) for w in w_cols]
    z_cols_used = [grouping_map.get(z, z) for z in z_cols]

    #the user chooses which values are x0, x1, and y (I need them for identification formulas)
    st.subheader("Protected Attribute and Outcome Setup")

    x0_value = None
    x1_value = None
    y_value = None

    if x_col_used is not None:
        x_unique = sorted(df[x_col_used].dropna().unique().tolist())
        st.markdown(f"Unique values in **X ({x_col_used})**: `{x_unique}`")
        if len(x_unique) > 0:
            x0_value = st.selectbox(
                "x0 (disadvantaged group)",
                options=x_unique,
                index=0,
            )
            #pick second distinct value as x1 if available
            default_idx = 1 if len(x_unique) > 1 else 0
            x1_value = st.selectbox(
                "x1 (advantaged group)",
                options=x_unique,
                index=default_idx,
            )

    if y_col_used is not None:
        y_unique = sorted(df[y_col_used].dropna().unique().tolist())
        st.markdown(f"Unique values in **Y ({y_col_used})**: `{y_unique}`")
        if len(y_unique) > 0:
            y_value = st.selectbox(
                "y (favourable outcome value, e.g. income > 50K = 1)",
                options=y_unique,
                index=min(1, len(y_unique) - 1) if len(y_unique) > 1 else 0,
            )

    user_explanation = st.text_area(
        "Explain the meaning of X, Y, W, Z (recommended if the variables are not self-explainatory)",
        placeholder=(
            "For example: X is race (0 = non-White, 1 = White), "
            "Y is income (1 = income > 50K), "
            "W mediators are education and experience, Z is a confounder..."
        ),
    )



    if st.button("Run causal fairness analysis"):
        if x_col_used is None or y_col_used is None:
            st.error("Please select at least X and Y.")
            return
        if x0_value is None or x1_value is None or y_value is None:
            st.error("Please specify x0, x1 and y values.")
            return

        with st.spinner("Computing causal fairness decomposition..."):
            try:
                results = run_analysis(
                    df=df,
                    x_col=x_col_used,
                    y_col=y_col_used,
                    w_cols=w_cols_used,
                    z_cols=z_cols_used,
                    x0_value=x0_value,
                    x1_value=x1_value,
                    y_value=y_value,
                    user_explanation=user_explanation,
                )
            except Exception as e:
                st.error(f"Error in compute_effects_multi / run_analysis: {e}")
                return

        st.session_state["results"] = results
        st.success("Analysis completed. See the results below.")
    

    #table with all effects
    if st.session_state["results"] is not None:
        results = st.session_state["results"]
        effects = results["effects"]

        st.subheader("Decomposition Table")
        rows = []

        main_keys = ["total_variation", "total_effect", "indirect_effect", "direct_effect", "spurious_effect"]
        for k in main_keys:
            if k in effects:
                rows.append({"Component": k, "Value": effects[k]})

        #with spurios decomp (if not empty)
        spurious = effects.get("spurious effect decomposition", {})
        if isinstance(spurious, dict) and len(spurious) > 0:
            for name, val in spurious.items():
                rows.append({"Component": f"spurious: {name}", "Value": val})

        #with indirect decomp (if not empty)
        indirect = effects.get("indirect effect decomposition", {})
        if isinstance(indirect, dict) and len(indirect) > 0:
            for name, val in indirect.items():
                rows.append({"Component": f"indirect: {name}", "Value": val})

        df_effects = pd.DataFrame(rows)
        st.dataframe(df_effects, use_container_width=True)


        #Interested in X-specific and Z-specific effects?
        st.subheader("Optional: detailed effects")

        #X-specific effects
        show_x_specific = st.checkbox("Show X-specific effects (for selected values of X)", value=False)
        if show_x_specific:
            #all possible X values in the *used* X column
            x_unique_used = sorted(df[x_col_used].dropna().unique().tolist())

            #try to preselect x0 and x1 if they exist in the data
            default_x = []
            for v in [x0_value, x1_value]:
                if v in x_unique_used and v not in default_x:
                    default_x.append(v)
            if not default_x and x_unique_used:
                default_x = [x_unique_used[0]]

            selected_x_vals = st.multiselect(
                f"Select X values for X-specific effects (column: {x_col_used})",
                options=x_unique_used,
                default=default_x,
            )

            df_cs = df.copy()
            x_rows = []

            for x_cond in selected_x_vals:
                try:
                    te_x, ie_x, de_x, se_x = compute_x_specific_effects(
                        df=df_cs,
                        x0=x0_value,
                        x1=x1_value,
                        x_cond=x_cond,      #effect for individuals with X = x_cond
                        y_val=y_value,
                        x_col=x_col_used,
                        y_col=y_col_used,
                        w_cols=w_cols_used,
                        z_cols=z_cols_used,
                    )
                except Exception as e:
                    st.error(f"Error computing X-specific effects for X={x_cond}: {e}")
                else:
                    x_rows.append({
                        "X_value": x_cond,
                        "total_effect_x":   te_x,
                        "indirect_effect_x": ie_x,
                        "direct_effect_x":   de_x,
                        "spurious_effect_x": se_x,
                    })

            if x_rows:
                st.markdown("**X-specific effects** for the selected X values:")
                df_x_spec = pd.DataFrame(x_rows)
                st.dataframe(df_x_spec, use_container_width=True)

        #Z-specific effects

        if z_cols_used:
            show_z_specific = st.checkbox("Show Z-specific effects by confounder combination", value=False)
            if show_z_specific:
                st.markdown(
                    f"Z-specific effects will be computed using these confounders: `{z_cols_used}`. "
                    "The function returns effects for all observed Z combinations; then you can pick one."
                )

                df_cs = df.copy()
                try:
                    effects_z = compute_z_specific_effects(
                        df=df_cs,
                        x0=x0_value,
                        x1=x1_value,
                        y_val=y_value,
                        x_col=x_col_used,
                        y_col=y_col_used,
                        w_cols=w_cols_used,
                        z_cols=z_cols_used,
                    )
                except Exception as e:
                    st.error(f"Error computing Z-specific effects: {e}")
                else:
                        rows = []
                        for key_tuple, effect_dict in effects_z.items():
                            row = {}

                                # Add Z-column values
                            for i, value in enumerate(key_tuple):
                                    row[f"Z{i+1}"] = value
                            row.update(effect_dict)

                            rows.append(row)

                        df_table = pd.DataFrame(rows)

                            
                        st.subheader("Z-specific Effects Table")
                        st.dataframe(df_table, use_container_width=True)

                                                    

        #LLM report (not fine-tuned)
        st.subheader("Generate textual fairness report")

        if st.button("Generate LLM report"):
            with st.spinner("Asking the LLM for a fairness-oriented explanation..."):
                try:
                    summary = summarize_with_llm(results)
                except Exception as e:
                    st.error(f"Error calling LLM: {e}")
                    return

            st.markdown(summary)




if __name__ == "__main__":
    main()





