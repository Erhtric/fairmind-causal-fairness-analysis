from pathlib import Path
from typing import Any
import json
import pandas as pd
from openai import BadRequestError


# ONLY LLM, no results
def summarize_with_llm_combined_dataset_only():
    """
    Dataset-only version:
    - Dataset is attached as an input file.
    - Model must compute fairness decomposition from dataset.
    Returns: (system_prompt, user_prompt)
    Column roles:
    - Y (outcome variable): <PUT COLUMN NAME>
    - Z (sensitive attribute): <PUT COLUMN NAME>
    - X (predictor features): <PUT COLUMN NAMES OR 'all except Y and Z'>
    - W (control variables, if any): <PUT COLUMN NAMES OR 'none'>

    """
    REPO_ROOT = Path(__file__).resolve().parents[1]
    prompt_path = Path(REPO_ROOT / "prompts" / "prompts_onlygpt.txt")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    user_prompt = """
        You are given a dataset as a PDF rendering of the dataset (table)."

        Column roles:
        - X (protected variable): <Sex>, with x0= Female and x1=Male
        - Y (outcome attribute): <Admission> y= 'Accepted'
        - Z (spurious features): <None>
        - W (mediator variables): <Major>


        Your task:
        1. Analyze the dataset.
        2. Compute the fairness decomposition according to Plecko and Bareinboim theory, both general and X-Z specific effects.
        3. Produce a structured report.

        Output format MUST be:

        TEXT:
        <plain language report>

        LATEX:
        <standalone LaTeX document>
        """

    return system_prompt, user_prompt


def generate_report_from_file_id(
    file_id: str,
    client,
    model: str = "gpt-5-mini",
    prompt_kwargs=None,
):
    prompt_kwargs = prompt_kwargs or {}

    system_prompt, user_prompt = summarize_with_llm_combined_dataset_only(
        **prompt_kwargs
    )

    print("USER_PROMPT:", user_prompt)
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "high"},
            instructions=system_prompt,
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto", "file_ids": [file_id]},
                },
            ],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                }
            ],
        )
    except BadRequestError as e:
        print("BadRequestError:\n", str(e))
        raise

    print("USAGE:", resp.usage)

    full_output = (resp.output_text or "").strip()

    if "LATEX:" not in full_output:
        raise ValueError(
            "Model output did not contain a LATEX section.\n\nOUTPUT:\n"
            + full_output[:1500]
        )

    text_part, latex_part = full_output.split("LATEX:", 1)
    return text_part.replace("TEXT:", "").strip(), latex_part.strip()


# LLM, with results
def prepare_llm_payload_general(
    *,
    dataset_name: str,
    X: str,
    Y: str,
    W=None,
    Z=None,
    x0=None,
    x1=None,
    y_target=None,
    results=None,
    stepwise_results=None,
    variable_metadata=None,
    cont_results=None,
    state_names=None,
    graph_edges=None,
    checks=None,
    notes=None,
    ie_decomp=None,
    se_decomp=None,
):
    W = [] if W is None else W
    Z = [] if Z is None else Z
    variable_metadata = {} if variable_metadata is None else variable_metadata
    checks = {} if checks is None else checks
    stepwise_results = [] if stepwise_results is None else stepwise_results
    cont_results = [] if cont_results is None else cont_results
    ie_decomp = [] if ie_decomp is None else ie_decomp
    se_decomp = [] if se_decomp is None else se_decomp
    if isinstance(results, pd.Series):
        results = results.to_dict()
    elif isinstance(results, pd.DataFrame):
        results = results.to_dict(orient="records")

    if isinstance(variable_metadata, pd.Series):
        variable_metadata = variable_metadata.to_dict()

    if isinstance(state_names, pd.Series):
        state_names = state_names.to_dict()

    payload = {
        "dataset": dataset_name,
        "causal_query": {
            "X": X,
            "Y": Y,
            "W": W,
            "Z": Z,
            "x0": x0,
            "x1": x1,
            "y_target": y_target,
        },
        "variable_metadata": variable_metadata,
        "state_names": state_names,
        "graph_edges": graph_edges,
        "results": results,
        "stepwise_results": stepwise_results,
        "continuous_results": cont_results,
        "ie_decomp": ie_decomp,
        "se_decomp": se_decomp,
        "checks": checks,
        "notes": notes,
    }

    return payload


def payload_to_json(payload: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False, default=str)


def summarize_with_llm_combined(results, client):
    REPO_ROOT = Path(__file__).resolve().parents[1]

    results_json = json.dumps(results, indent=2)

    prompt_path = Path(REPO_ROOT / "prompts" / "fairmind_prompts.txt")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    user_prompt = f"""
Here are the fairness decomposition results in JSON:

{results_json}
"""

    resp = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "high"},
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        ],
    )

    full_output = (resp.output_text or "").strip()

    # Check if there's latex
    if "LATEX:" not in full_output:
        raise ValueError("LLM output did not contain LATEX section.")

    text_part, latex_part = full_output.split("LATEX:", 1)
    text_part = text_part.replace("TEXT:", "").strip()
    latex_part = latex_part.strip()

    token_usage = resp.usage

    return text_part, latex_part, token_usage
