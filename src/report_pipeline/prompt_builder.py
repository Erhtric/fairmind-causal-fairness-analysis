"""Costruzione del prompt per la generazione del report LaTeX qualitativo.

Implementa i punti 1-3 del feedback del docente: i cinque effetti causali
sono calcolati esclusivamente da FairMind (mai dall'LLM), inseriti in un
template LaTeX a struttura rigida, e il modello ha la sola responsabilita'
di riempire i blocchi di interpretazione qualitativa e la sezione finale
"Recap Questions". Il template NON viene mai lasciato scrivere all'LLM per
intero: solo i segnaposto ``<<...>>`` rimasti dopo il pre-riempimento dei
dati numerici (fatto qui, in Python, non dal modello) sono editabili.
"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "template.tex"

# Segnaposto che lo script pre-riempie con i dati esatti di FairMind, PRIMA
# di mandare il template all'LLM. Il modello non li vede mai come vuoti.
_METADATA_KEYS = [
    "REPORT_DATE", "DATASET", "PROTECTED_ATTR", "X0", "X1",
    "OUTCOME_ATTR", "MEDIATOR", "CONFOUNDER",
]
_EFFECT_KEYS = ["TV", "TE", "SE", "DE", "IE"]

# Segnaposto che restano nel template dopo il pre-riempimento: solo questi
# sono responsabilita' dell'LLM.
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
    """Escapa i caratteri speciali di LaTeX in un valore inserito a runtime.

    I nomi di colonna del dataset (es. ``S2_gender``, ``T_income``) contengono
    spesso underscore, che in LaTeX aprono un pedice fuori da modalita'
    matematica e mandano in errore la compilazione (``Missing $ inserted``).
    Va applicato a qualunque metadato che finisce nel template grezzo, non ai
    valori numerici degli effetti (gia' formattati come float, nessun
    carattere speciale possibile).
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
    """Sostituisce SOLO i segnaposto di metadati e valori numerici.

    I segnaposto qualitativi (``QUALITATIVE_*``) e le risposte del recap
    (``ANSWER_Q*``) restano intatti: sono quelli che l'LLM dovra' riempire.
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
   effect that passes through the mediator W. Explain whether the mediator \
   channel amplifies or offsets (mitigates) the direct effect, based on \
   whether IE and DE have the same or opposite sign.
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
    """Punto di ingresso principale: ritorna (system_prompt, user_prompt).

    ``effects`` deve avere le chiavi TV, TE, SE, DE, IE (float).
    ``context`` deve avere le chiavi dataset, protected_attr, x0, x1,
    outcome_attr, e opzionalmente mediator, confounder.
    """
    template = load_template()
    prefilled = prefill_template(template, effects, context, report_date)
    return SYSTEM_PROMPT, build_user_prompt(prefilled)
