from pathlib import Path

from src.visualisation.latex import DEFAULT_TEMPLATE_PATH, fill_sections_latex_template


def test_template_file_exists():
    """Verify that the default LaTeX template file exists."""
    assert DEFAULT_TEMPLATE_PATH.exists(), (
        f"Template file not found at {DEFAULT_TEMPLATE_PATH}"
    )


def test_render_latex_template_basic():
    """Verify that section content is properly injected into the LaTeX template."""
    overview = "This is the overview paragraph."
    decomp = "Bullet 1: Total Variation is 0.15.\nBullet 2: Total Effect is 0.10."
    eval_section = "1. [EQ_QUAL_01]: Passed."
    recap = "Recap paragraph confirming Female disparity."

    rendered = fill_sections_latex_template(
        overview=overview,
        decomposition=decomp,
        logical_eval=eval_section,
        recap=recap,
        dataset_name="Adult Census Dataset",
    )

    assert "\\documentclass" in rendered
    assert (
        "\\title{\\textbf{Fairness Decomposition Report: Adult Census Dataset}}"
        in rendered
    )
    assert overview in rendered
    assert decomp in rendered
    assert eval_section in rendered
    assert recap in rendered
    assert "{{ OVERVIEW_SECTION }}" not in rendered
    assert "{{ DECOMPOSITION_SECTION }}" not in rendered


def test_fallback_template():
    """Verify fallback rendering when template file path is invalid."""
    rendered = fill_sections_latex_template(
        overview="Overview text",
        decomposition="Decomp text",
        logical_eval="Eval text",
        recap="Recap text",
        dataset_name="Synthetic Data",
        template_path=Path("/nonexistent/path/template.tex"),
    )

    assert "\\documentclass" in rendered
    assert "Synthetic Data" in rendered
    assert "Overview text" in rendered


def test_latex_to_plain_text():
    """Verify that LaTeX syntax is converted into clean plain text / Markdown."""
    from src.visualisation.latex import latex_to_plain_text

    latex = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\subsection*{1. Overview of the Fairness Analysis}\n"
        "This compares $X = \\text{Female}$ vs $X = \\text{Male}$ with outcome $\\mathrm{T\\_grade} \\geq 50K$.\n"
        "\\subsection*{2. Decomposition of Effects}\n"
        "\\begin{itemize}\n"
        "  \\item \\textbf{Total Variation (TV):} Value is 0.1500 (approx \\approx 15\\%).\n"
        "\\end{itemize}\n"
        "\\subsection*{Recap}\n"
        "Conclusion statement.\n"
        "\\end{document}"
    )

    text = latex_to_plain_text(latex)
    assert "\\documentclass" not in text
    assert "\\begin{document}" not in text
    assert "### Overview of the Fairness Analysis" in text
    assert "T_grade >= 50K" in text
    assert "**Total Variation (TV):**" in text
    assert "15%" in text
    assert "### Recap" in text
