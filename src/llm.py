
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd

# ONLY LLM, no results


def summarize_with_llm_combined_dataset_only(
    x: str,
    y: str,
    w: list[str] | None = None,
    z: list[str] | None = None,
    x0: str | None = None,
    x1: str | None = None,
    y_value: str | None = None,
    w_values: list[str] | None = None,
    prompt_filename: str = "prompts_onlygpt.txt",
):
    """
    Generic dataset-only prompt builder.

    Parameters
    ----------
    x : protected variable
    y : outcome variable
    w : mediator variables
    z : spurious features
    x0, x1 : reference values for X
    y_value : target value for Y
    w_values : optional allowed values for W
    prompt_filename : prompt template file name

    Returns
    -------
    (system_prompt, user_prompt)
    """
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / prompt_filename
    system_prompt = prompt_path.read_text(encoding="utf-8")

    z_text = ", ".join(z) if z else "None"
    w_text = ", ".join(w) if w else "None"
    w_values_text = list(w_values) if w_values else "not specified"

    x_line = f"- X (protected variable): <{x}>"
    if x0 is not None and x1 is not None:
        x_line += f", with x0={x0} and x1={x1}"

    y_line = f"- Y (outcome attribute): <{y}>"
    if y_value is not None:
        y_line += f" y='{y_value}'"

    z_line = f"- Z (spurious features): <{z_text}>"
    w_line = f"- W (mediator variables): <{w_text}>"
    if w and w_values is not None:
        w_line += f" {list(w_values_text)}"

    user_prompt = f"""
You are given a dataset as a PDF rendering of the dataset (table).

Column roles:
{x_line}
{y_line}
{z_line}
{w_line}

Your task:
1. Analyze the dataset.
2. Compute the fairness decomposition according to Bareinboim and Plecko theory, both general and X-Z specific effects.
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

    system_prompt, user_prompt = summarize_with_llm_combined_dataset_only(**prompt_kwargs)

    print("USER_PROMPT:", user_prompt)
    try:
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": [file_id],
                    },
                },
            ],
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                ],
            }],
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

from typing import Any, Optional
import pandas as pd

from typing import Any, Optional, List, Dict, Tuple
import pandas as pd


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
    state_names=None,
    graph_edges=None,
    checks=None,
    notes=None,
):
    W = [] if W is None else W
    Z = [] if Z is None else Z
    variable_metadata = {} if variable_metadata is None else variable_metadata
    checks = {} if checks is None else checks
    stepwise_results = [] if stepwise_results is None else stepwise_results

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
        "checks": checks,
        "notes": notes,
    }

    return payload

def payload_to_json(payload: Dict[str, Any], indent: int = 2) -> str:
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
        instructions=system_prompt,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt}
            ],
        }],
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