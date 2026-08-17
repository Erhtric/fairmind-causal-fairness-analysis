import os
import re
import shutil
import tempfile
from pathlib import Path

from loguru import logger
from pdflatex import PDFLaTeX

from src.llm import LLMReportResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "templates" / "report_template.tex"


def fill_sections_latex_template(
    overview: str,
    decomposition: str,
    logical_eval: str,
    recap: str,
    dataset_name: str = "Fairness Query",
    template_path: str | Path | None = None,
) -> str:
    """Fills LLM-generated section content into the structured LaTeX template.

    Args:
        overview: Content for Section 1 (Overview).
        decomposition: Content for Section 2 (Decomposition of Effects).
        logical_eval: Content for Section 3 (Logical Soundness Evaluation).
        recap: Content for the Recap section.
        dataset_name: Name of the dataset for the report title.
        template_path: Path to the LaTeX template file. Defaults to templates/report_template.tex.

    Returns:
        A complete, standalone LaTeX document string.
    """
    path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
    if not path.exists():
        logger.warning(f"Template file not found at {path}. Using fallback skeleton.")
        return _fallback_latex_template(
            overview, decomposition, logical_eval, recap, dataset_name
        )

    template_content = path.read_text(encoding="utf-8")

    rendered = (
        template_content.replace("{{ DATASET_NAME }}", dataset_name)
        .replace("{{ OVERVIEW_SECTION }}", overview.strip())
        .replace("{{ DECOMPOSITION_SECTION }}", decomposition.strip())
        .replace("{{ LOGICAL_SOUNDNESS_SECTION }}", logical_eval.strip())
        .replace("{{ RECAP_SECTION }}", recap.strip())
    )

    return rendered


def parse_latex_sections(latex_text: str) -> tuple[str, str, str, str]:
    """Parses individual section strings from a LaTeX report document.

    Returns:
        Tuple of (overview, decomposition, logical_eval, recap).
    """
    sec1_match = re.search(
        r"\\subsection\*?\{.*?Overview.*?\}(.*?)(?=\\subsection\*?\{|\\end\{document\}|$)",
        latex_text,
        re.DOTALL | re.IGNORECASE,
    )
    sec2_match = re.search(
        r"\\subsection\*?\{.*?Decomposition.*?\}(.*?)(?=\\subsection\*?\{|\\end\{document\}|$)",
        latex_text,
        re.DOTALL | re.IGNORECASE,
    )
    sec3_match = re.search(
        r"\\subsection\*?\{.*?Logical.*?\}(.*?)(?=\\subsection\*?\{|Recap|\\end\{document\}|$)",
        latex_text,
        re.DOTALL | re.IGNORECASE,
    )
    recap_match = re.search(
        r"\\subsection\*?\{.*?Recap.*?\}(.*?)(?=\\end\{document\}|$)",
        latex_text,
        re.DOTALL | re.IGNORECASE,
    )

    overview = sec1_match.group(1).strip() if sec1_match else ""
    decomposition = sec2_match.group(1).strip() if sec2_match else ""
    logical_eval = sec3_match.group(1).strip() if sec3_match else ""
    recap = recap_match.group(1).strip() if recap_match else ""

    return overview, decomposition, logical_eval, recap


def normalize_latex_causal_notation(latex_text: str) -> str:
    """Cleans up common non-standard causal notations (e.g., converts Y_{do(X=x_1)} -> Y_{x_1})."""
    if not latex_text:
        return ""
    # Convert Y_{do(X=x_1)} -> Y_{x_1}
    normalized = re.sub(
        r"Y_\{do\([A-Za-z0-9\_]*\s*=\s*([^\}]+)\)\}", r"Y_{\1}", latex_text
    )
    normalized = re.sub(r"Y_\{do\((.*?)\)\}", r"Y_{\1}", normalized)
    return normalized


def ensure_template_rendered_latex(
    latex_text: str, dataset_name: str = "Fairness Query"
) -> str:
    """Parses raw LLM LaTeX output and renders it via fill_sections_latex_template."""
    latex_text = normalize_latex_causal_notation(latex_text)
    overview, decomposition, logical_eval, recap = parse_latex_sections(latex_text)
    if overview or decomposition or recap:
        return fill_sections_latex_template(
            overview=overview,
            decomposition=decomposition,
            logical_eval=logical_eval,
            recap=recap,
            dataset_name=dataset_name,
        )
    return latex_text


def _fallback_latex_template(
    overview: str,
    decomposition: str,
    logical_eval: str,
    recap: str,
    dataset_name: str,
) -> str:
    """Fallback inline template if file is missing."""
    return f"""\\documentclass[11pt, a4paper]{{article}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{booktabs}}
\\usepackage{{parskip}}

\\title{{\\textbf{{Fairness Decomposition Report: {dataset_name}}}}}
\\author{{Causal Fairness Analysis Engine}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\subsection*{{1. Overview of the Fairness Analysis}}
{overview.strip()}

\\subsection*{{2. Decomposition of Effects}}
{decomposition.strip()}

\\subsection*{{3. Logical Soundness Evaluation}}
{logical_eval.strip()}

\\subsection*{{Recap}}
{recap.strip()}

\\end{{document}}"""


def compile_report_to_pdf(
    report: LLMReportResult | str, output_path: str | Path
) -> None:
    """Compiles the LaTeX report to a PDF file.

    Args:
        report: An LLMReportResult instance or raw LaTeX string.
        output_path: The output path where the PDF will be saved.
    """
    latex_content = report.latex if isinstance(report, LLMReportResult) else report
    latex_content = ensure_template_rendered_latex(latex_content)

    temp_dir = tempfile.mkdtemp(prefix="latex_report_")

    try:
        temp_tex_path = os.path.join(temp_dir, "report.tex")
        with open(temp_tex_path, "w", encoding="utf-8") as f:
            logger.info(f"Writing LaTeX report to temporary file: {temp_tex_path}")
            f.write(latex_content)

        logger.info(f"Creating PDF from LaTeX file: {temp_tex_path}")
        pdfl = PDFLaTeX.from_texfile(temp_tex_path)
        pdf, _, _ = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)

        output_path = Path(output_path)
        if output_path.exists():
            logger.info(f"Removing existing PDF report at: {output_path}")
            output_path.unlink()

        with open(output_path, "wb") as f:
            logger.info(f"Saving PDF report to: {output_path}")
            f.write(pdf)

        logger.info(f"PDF report generated successfully at: {output_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def latex_to_plain_text(latex_text: str) -> str:
    """Converts a LaTeX report document into clean plain text / Markdown for display."""
    if not latex_text:
        return ""

    text = str(latex_text)

    # Strip comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # Strip preamble and document tags
    text = re.sub(r"\\documentclass.*?\\begin\{document\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\end\{document\}", "", text)
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\date\{.*?\}", "", text)
    text = re.sub(r"\\author\{.*?\}", "", text)
    text = re.sub(r"\\title\{.*?\}", "", text)

    # Convert headings
    text = re.sub(r"\\subsection\*?\{(?:\d+\.\s*)?(.*?)\}", r"\n\n### \1\n", text)
    text = re.sub(r"\\section\*?\{(.*?)\}", r"\n\n## \1\n", text)

    # Convert lists
    text = re.sub(r"\\begin\{itemize\}", "", text)
    text = re.sub(r"\\end\{itemize\}", "", text)
    text = re.sub(r"\\begin\{enumerate\}(?:\[.*?\])?", "", text)
    text = re.sub(r"\\end\{enumerate\}", "", text)
    text = re.sub(r"\\item\s+", r"\n- ", text)

    # Convert text styles
    text = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{(.*?)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{(.*?)\}", r"`\1`", text)
    text = re.sub(r"\\mathrm\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)

    # Math operators & symbols
    text = re.sub(r"\\geq", r">=", text)
    text = re.sub(r"\\leq", r"<=", text)
    text = re.sub(r"\\neq", r"!=", text)
    text = re.sub(r"\\times", r"*", text)
    text = re.sub(r"\\approx", r"≈", text)
    text = re.sub(r"\\mid", r"|", text)

    # Unescape special characters
    text = text.replace(r"\_", "_")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\$", "$")

    # Clean double backslashes
    text = re.sub(r"\\\\\s*", "\n", text)

    # Clean up empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
