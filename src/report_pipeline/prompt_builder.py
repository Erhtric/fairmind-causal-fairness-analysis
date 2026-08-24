"""Prompt construction for the qualitative LaTeX report.

The five effects are computed by FairMind alone and written into a rigid LaTeX
template, so the model is only responsible for the qualitative blocks and the
Recap Questions. The model never writes the template: the numbers are filled
in here, in Python, and only the placeholders left after that are editable.
"""

from pathlib import Path

from .validator import EPS_IE, THRESH_DE, THRESH_SE_REL, THRESH_TV

TEMPLATE_PATH = Path(__file__).parent / "template.tex"

# Placeholders prefilled with the exact FairMind values before the template
# reaches the model, which therefore never sees them empty.
#
# The names below must appear in the template exactly as written, underscore
# included and not escaped. The token is replaced before LaTeX ever sees it,
# so it is not LaTeX text; the value going into it is what gets escaped, by
# escape_latex(). Writing ``<<PROTECTED\_ATTR>>`` in the template turns the
# replace() into a silent no-op: the value is computed, discarded, and the
# placeholder reaches the model, which fills it by inventing. That happened
# with the English translation. tests/test_report_pipeline_prompt.py pins the
# contract.
_METADATA_KEYS = [
    "REPORT_DATE", "DATASET", "PROTECTED_ATTR", "X0", "X1",
    "OUTCOME_ATTR", "MEDIATOR", "CONFOUNDER",
]
_EFFECT_KEYS = ["TV", "TE", "SE", "DE", "IE"]

# Placeholders filled with our own text rather than with data: they carry
# deliberate math mode and must go in unescaped.
_LITERAL_KEYS = ["THRESHOLDS"]


def thresholds_note() -> str:
    """The line that states, inside the report, the thresholds behind the answers.

    Without it a reader cannot tell why an answer is YES rather than NO: the
    thresholds lived only in the validator, and the question that made them
    necessary came from someone reading a report without the code at hand.
    """
    return (
        "The recap answers follow deterministic threshold rules applied to the "
        "values above: direct discrimination when $|\\mathrm{DE}| > "
        f"{THRESH_DE}$; a mediator channel counted as negligible when "
        f"$|\\mathrm{{IE}}| < {EPS_IE}$; practical relevance when "
        f"$|\\mathrm{{TV}}| > {THRESH_TV}$; a substantial spurious component "
        f"when $|\\mathrm{{SE}}| / |\\mathrm{{TV}}| > {THRESH_SE_REL}$."
    )

# Placeholders surviving the prefill: these alone are the model's job.
QUALITATIVE_PLACEHOLDERS = ["QUALITATIVE_TOTAL", "QUALITATIVE_DE", "QUALITATIVE_IE"]
ANSWER_PLACEHOLDERS = ["ANSWER_Q1", "ANSWER_Q2", "ANSWER_Q3", "ANSWER_Q4", "ANSWER_Q5"]


_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in a value injected at runtime.

    Column names such as ``S2_gender`` carry underscores, which outside math
    mode open a subscript and break compilation with ``Missing $ inserted``.
    Apply to any metadata reaching the raw template, but not to the effects,
    already formatted as floats.
    """
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in text)


def load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def prefill_template(
    template: str,
    effects: dict[str, float],
    context: dict[str, str],
    report_date: str,
) -> str:
    """Replace only the metadata, numeric and literal placeholders.

    ``QUALITATIVE_*`` and ``ANSWER_Q*`` are left untouched: those are the ones
    the model has to fill.
    """
    filled = template
    values = {
        "REPORT_DATE": report_date,
        "DATASET": context["dataset"],
        "PROTECTED_ATTR": context["protected_attr"],
        "X0": context["x0"],
        "X1": context["x1"],
        "OUTCOME_ATTR": context["outcome_attr"],
        "MEDIATOR": context.get("mediator", "--"),
        "CONFOUNDER": context.get("confounder", "--"),
    }
    for key in _METADATA_KEYS:
        filled = filled.replace(f"<<{key}>>", escape_latex(str(values[key])))
    for key in _EFFECT_KEYS:
        filled = filled.replace(f"<<{key}>>", f"{effects[key]:.6f}")
    literals = {"THRESHOLDS": thresholds_note()}
    for key in _LITERAL_KEYS:
        filled = filled.replace(f"<<{key}>>", literals[key])
    return filled


SYSTEM_PROMPT = """You are a causal fairness reporting assistant. Your ONLY task \
is to fill in a small number of designated placeholders inside a fixed LaTeX \
document. You are NOT a calculator: the five causal fairness effects (TV, TE, \
SE, DE, IE) have already been computed exactly by FairMind, a deterministic \
Bayesian-network inference engine, and are already written into the document \
you receive. Do not recompute, question, or restate them as if you derived them.

You will receive the full LaTeX source of the report as it currently stands. \
It contains exactly eight placeholders, each written as a literal token of the \
form <<NAME>>. Every other character in the document — preamble, packages, \
section and subsection headings, the configuration table, the effects table, \
the enumerate list, the closing \\end{document} — is FINAL and must be \
reproduced byte-for-byte in your output. Do not add, remove, reorder, or \
rename any LaTeX command, environment, section, or package. Do not add your \
own commentary outside the placeholders.

The eight placeholders and what belongs in each:

1. <<QUALITATIVE_TOTAL>> — 3-5 sentences interpreting TV, TE and SE together: \
   how large is the total disparity between the two groups, how much of it \
   survives as a genuine causal effect (TE) once confounding is removed, and \
   how much was spurious (SE, driven by the confounder Z rather than by X \
   itself).
2. <<QUALITATIVE_DE>> — 3-5 sentences interpreting DE: the portion of the \
   effect that goes directly from X to Y, not through the mediator. State \
   in plain language whether this looks like direct discrimination against \
   the protected group, and how large it is relative to TE.
3. <<QUALITATIVE_IE>> — 3-5 sentences interpreting IE: the portion of the \
   effect that passes through the mediator W, shown here so that TE = DE + IE \
   (a positive IE adds to the direct effect; a negative one subtracts from \
   it). Explain whether the mediator channel amplifies the direct effect \
   (IE has the SAME sign as DE, so |TE| > |DE|) or offsets it, i.e. \
   mitigates it (IE has the OPPOSITE sign to DE, so |TE| < |DE|).
4. <<ANSWER_Q1>> through <<ANSWER_Q5>> — the boolean answer to each Recap \
   Question, using EXACTLY the token YES or NO (uppercase, no \
   punctuation, nothing else). Base each answer strictly on the numeric \
   values given (DE, IE, TV, SE) and on the qualitative text you just wrote \
   for that section — do not introduce new claims in the recap that \
   contradict what you wrote above.

Rules:
- Write the qualitative blocks in ENGLISH, in plain and precise language \
  grounded only in the numbers provided. The whole document is in English: \
  do not switch language. Do not invent context about the dataset that is not \
  given to you.
- Never leave a placeholder token (<<...>>) in your output — every one of \
  the eight must be replaced.
- Never touch text outside the eight placeholders.
- Output ONLY the complete LaTeX document, starting with \\documentclass and \
  ending with \\end{document}. No markdown code fences, no explanations \
  before or after.
"""


def build_user_prompt(prefilled_template: str) -> str:
    return f"""Here is the report template with the FairMind configuration and \
the five exact causal fairness effects already filled in. Fill in the eight \
remaining <<...>> placeholders as instructed, and return the complete LaTeX \
document.

{prefilled_template}
"""


def build_prompts(
    effects: dict[str, float],
    context: dict[str, str],
    report_date: str,
) -> tuple[str, str]:
    """Main entry point: returns ``(system_prompt, user_prompt)``.

    ``effects`` needs the keys TV, TE, SE, DE, IE. ``context`` needs dataset,
    protected_attr, x0, x1 and outcome_attr, with mediator and confounder
    optional.
    """
    template = load_template()
    prefilled = prefill_template(template, effects, context, report_date)
    return SYSTEM_PROMPT, build_user_prompt(prefilled)
