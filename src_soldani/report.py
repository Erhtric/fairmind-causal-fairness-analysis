import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.graph import filter_nodes_by_type
from src.visualisation.sankey import plot_effect_sankey_percent


def generate_html_report(
    dataset_name: str,
    config: dict,
    manual_effects: dict[str, float] | None,
    learned_effects: dict[str, float] | None,
    llm_results: dict[str, dict[str, float]] | None,
    intersectional_results: dict[str, dict[str, float]] | None,
    discrepancies: pd.DataFrame | None,
    similarity_metrics: dict[str, float] | None,
    timing: dict[str, float],
    output_path: str | Path = "benchmark_results/multimodal_report.html",
):
    """Generate a comprehensive HTML report with all results.

    Parameters
    ----------
    dataset_name : str
    config : dict
        Original configuration.
    manual_effects : dict or None
        FairMind effects on the manually specified graph.
    learned_effects : dict or None
        FairMind effects on the discovered graph.
    llm_results : dict or None
        {model_name: {TV: ..., TE: ..., SE: ..., DE: ..., IE: ...}}.
    intersectional_results : dict or None
        {(x0, x1): {TV: ..., ...}}.
    discrepancies : pd.DataFrame or None
        Columns: effect, fairmind, llm, abs_error, rel_error_% (per model).
    similarity_metrics : dict or None
        From graph_similarity().
    timing : dict
        {phase_name: seconds}.
    output_path : str
        Where to save the HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    def _h(text, level=2):
        return f"<h{level}>{text}</h{level}>"

    def _p(text):
        return f"<p>{text}</p>"

    def _table(df, cls="dataframe"):
        if df is None or len(df) == 0:
            return _p("<em>No data</em>")
        return df.to_html(index=False, classes=cls, float_format="%.6f")

    sections.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>FairMind — Multimodal Report ({dataset_name})</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1200px; margin: 2em auto; padding: 0 1em;
                background: #fafafa; color: #222;
            }}
            h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 0.3em; }}
            h2 {{ color: #16213e; margin-top: 2em; }}
            h3 {{ color: #0f3460; }}
            table.dataframe {{
                border-collapse: collapse; width: 100%; margin: 1em 0;
                background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            table.dataframe th {{
                background: #16213e; color: white; padding: 8px 12px; text-align: left;
            }}
            table.dataframe td {{
                padding: 6px 12px; border-bottom: 1px solid #eee;
            }}
            table.dataframe tr:hover {{ background: #f0f0f5; }}
            .plot-container {{ margin: 2em 0; text-align: center; }}
            .badge {{
                display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 0.85em; font-weight: 600;
            }}
            .badge-green {{ background: #d4edda; color: #155724; }}
            .badge-yellow {{ background: #fff3cd; color: #856404; }}
            .badge-red {{ background: #f8d7da; color: #721c24; }}
            .footer {{ margin-top: 3em; font-size: 0.8em; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>FairMind — Causal Fairness Analysis Report</h1>
        <p><strong>Dataset:</strong> {dataset_name} &nbsp;|&nbsp;
           <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    """)

    sections.append(_h("Configuration", 2))
    config_html = "<table class='dataframe'><tr><th>Parameter</th><th>Value</th></tr>"
    for k, v in config.items():
        config_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    config_html += "</table>"
    sections.append(config_html)

    if similarity_metrics is not None:
        sections.append(_h("Graph Discovery: Expert vs Learned", 2))
        sections.append(_table(pd.DataFrame([similarity_metrics])))
        secs = timing.get("discovery", 0)
        sections.append(_p(f"Discovery time: {secs:.4f}s"))

    if manual_effects is not None and learned_effects is not None:
        sections.append(_h("FairMind: Manual Graph vs Discovered Graph", 2))
        comp = pd.DataFrame([manual_effects, learned_effects], index=["Manual", "Learned"])
        sections.append(_table(comp.transpose().reset_index().rename(columns={"index": "Effect"})))

    if manual_effects is not None:
        sections.append(_h("FairMind Ground Truth (Manual Graph)", 2))
        sections.append(_table(pd.DataFrame([manual_effects])))

    if learned_effects is not None:
        sections.append(_h("FairMind Ground Truth (Discovered Graph)", 2))
        sections.append(_table(pd.DataFrame([learned_effects])))

    if llm_results is not None and len(llm_results) > 0:
        sections.append(_h("Multi-Model Benchmark", 2))
        rows = []
        for model_name, effs in llm_results.items():
            rows.append({"Model": model_name, **{k: round(v, 6) for k, v in effs.items()}})
        sections.append(_table(pd.DataFrame(rows)))

        if discrepancies is not None:
            sections.append(_h("Discrepancies FairMind vs LLM", 2))
            sections.append(_table(discrepancies))

    if intersectional_results is not None and len(intersectional_results) > 0:
        sections.append(_h("Intersectional Analysis", 2))
        rows = []
        for key, effs in intersectional_results.items():
            rows.append({"Pair": key, **{k: round(v, 6) for k, v in effs.items()}})
        df_inter = pd.DataFrame(rows)
        sections.append(_table(df_inter))

    sections.append(_h("Timing", 2))
    timing_df = pd.DataFrame([timing])
    sections.append(_table(timing_df))

    sections.append("""
        <div class="footer">
            Generated by FairMind — Soldani Causal Fairness Analysis
        </div>
    </body></html>
    """)

    html = "\n".join(sections)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report salvato: {output_path}")
    return str(output_path)
