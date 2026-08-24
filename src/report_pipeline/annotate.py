"""Annotate a generated report with the expected answers.

The expected answers cannot appear in the template the model receives: it
would read them and copy them, and the score would stop measuring anything.

This module therefore runs downstream, on a document already generated and
already scored. It rewrites the Recap Questions section, putting each model
answer next to the expected one and the outcome of the comparison, and appends
a summary line with the score. The original file stays on disk as evidence of
what the model actually wrote.
"""

from __future__ import annotations

import re

from .validator import ANSWER_LINE_PATTERN, ScoreReport

# The pattern is not redefined here: it is the one the validator uses to read
# the answers. Pairing with score.results happens by position, so both modules
# have to find the same occurrences in the same order. With two diverging
# copies, a document where only some answers are formatted differently would
# shift the pairing, and every question would show the expected answer of
# another one.
_ANSWER_LINE = ANSWER_LINE_PATTERN

_MISSING = "--"


def _verdict(correct: bool, llm_answer: str | None) -> str:
    if llm_answer is None:
        return "unreadable"
    return "correct" if correct else "wrong"


def annotate_recap_answers(latex_text: str, score: ScoreReport) -> str:
    """Put the expected answer and the outcome next to each model answer.

    Replaces the answer lines in order. The occurrences appear in the document
    in the same order the validator read them, so the pairing with
    ``score.results`` holds.

    If the document carries more answers than were scored, the surplus is left
    untouched: a malformed document must not be quietly tidied up.
    """
    results = list(score.results)

    def _replace(match: re.Match) -> str:
        if not results:
            return match.group(0)
        r = results.pop(0)
        llm = r.llm_answer if r.llm_answer is not None else _MISSING
        return (
            f"\\textbf{{LLM:}} {llm} \\quad "
            f"\\textbf{{Expected:}} {r.ground_truth} \\quad "
            f"({_verdict(r.correct, r.llm_answer)})"
        )

    annotated = _ANSWER_LINE.sub(_replace, latex_text)
    return _append_summary(annotated, score)


def _append_summary(latex_text: str, score: ScoreReport) -> str:
    """Append a line with the overall score after the list.

    It goes immediately before ``\\end{document}``, the only anchor that is
    guaranteed to be there: everything inside depends on what the model wrote.
    """
    summary = (
        "\n\\vspace{1em}\n"
        "\\noindent\\textbf{Answer consistency:} "
        f"{score.n_correct}/{score.n_total} "
        f"({score.score * 100:.1f}\\%). "
        "The expected answers are derived deterministically from the FairMind "
        "values alone, through threshold rules, and were added to the document "
        "after it was generated: the model never saw them.\n"
    )
    marker = "\\end{document}"
    if marker in latex_text:
        return latex_text.replace(marker, summary + "\n" + marker)
    return latex_text + summary
