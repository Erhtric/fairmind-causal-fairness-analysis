"""Parser and scorer for the generated LaTeX reports.

The report ends with five yes/no Recap Questions, answered by the model from
its own reading of the numbers. This module:

1. parses the five answers out of the LaTeX, by regex, without compiling it;
2. computes the expected answers from the FairMind values alone, through the
   threshold rules declared at the top of this file;
3. compares the two and returns a consistency score.

Usage:
    python -m src.report_pipeline.validator report.tex --effects effects.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field

###############################################################################
# Thresholds for the expected answers. Change them here.
###############################################################################

THRESH_DE = 0.05        # |DE| above this => direct discrimination
EPS_IE = 0.005          # |IE| below this => negligible channel
THRESH_TV = 0.05        # |TV| above this => practically relevant
THRESH_SE_REL = 0.10    # |SE|/|TV| above this => substantial spurious part

# The paper defines two distinct indirect effects, and they are not each
# other's opposite. Q2 and Q5 concern the decomposition TE = DE - IE, so they
# need the reverse form; the direct one flips the answer to Q2 with no visible
# symptom. The key is named explicitly, and its absence is an error rather
# than a default.
IE_DECOMPOSITION_KEY = "IE_reverse"


def decomposition_ie(effects: dict[str, float]) -> float:
    """The indirect effect in the form that enters the decomposition.

    Raises if the key is missing rather than falling back on "IE": a silent
    fallback would produce a wrong expected answer for Q2, and the score would
    look exactly the same.
    """
    if IE_DECOMPOSITION_KEY not in effects:
        raise KeyError(
            f"'{IE_DECOMPOSITION_KEY}' missing from the effects. The expected "
            "answers concern the decomposition TE = DE - IE and need the "
            "reverse form of the indirect effect; the direct one would change "
            "the answer to Q2."
        )
    return effects[IE_DECOMPOSITION_KEY]


###############################################################################
# Expected answers: one function per question, in template order.
###############################################################################

def gt_q1_direct_discrimination(effects: dict[str, float]) -> bool:
    """Q1: is there direct discrimination? DE large enough in magnitude."""
    return abs(effects["DE"]) > THRESH_DE


def gt_q2_ie_mitigates_te(effects: dict[str, float]) -> bool:
    """Q2: does the mediated effect mitigate the total, or amplify it?

    Write IE_add := -IE_{x1,x0}, so that TE = DE + IE_add. Same sign as DE
    means the mediator amplifies (|TE| > |DE|); opposite sign means it
    mitigates (|TE| < |DE|).

    decomposition_ie() returns IE_{x1,x0}, which is IE_add negated, so
    mitigation shows up here as a positive product with DE.

    On Adult: decomposition_ie = -0.046333 against DE = +0.138404, a negative
    product, so this returns False, meaning the mediator amplifies. Checked
    numerically and not only on paper, because an earlier version of this
    rule had the direction reversed.
    """
    ie, de = decomposition_ie(effects), effects["DE"]
    if abs(ie) < EPS_IE:
        return False
    return (ie * de) > 0


def gt_q3_tv_practically_relevant(effects: dict[str, float]) -> bool:
    """Q3: does TV clear the practical relevance threshold?"""
    return abs(effects["TV"]) > THRESH_TV


def gt_q4_se_substantial(effects: dict[str, float]) -> bool:
    """Q4: is the spurious part substantial against the observed total?"""
    tv, se = effects["TV"], effects["SE"]
    if abs(tv) < 1e-9:
        return abs(se) > 0
    return abs(se) / abs(tv) > THRESH_SE_REL


def gt_q5_de_dominant_over_ie(effects: dict[str, float]) -> bool:
    """Q5: does the direct component dominate the indirect one?"""
    return abs(effects["DE"]) > abs(decomposition_ie(effects))


# Order matches the questions in src/report_pipeline/template.tex.
GROUND_TRUTH_RULES = [
    gt_q1_direct_discrimination,
    gt_q2_ie_mitigates_te,
    gt_q3_tv_practically_relevant,
    gt_q4_se_substantial,
    gt_q5_de_dominant_over_ie,
]

N_QUESTIONS = len(GROUND_TRUTH_RULES)


###############################################################################
# Parsing the yes/no answers out of the generated LaTeX.
###############################################################################

# Captures everything after the label up to end of line, leaving
# _normalize_answer to isolate the answer.
#
# The earlier pattern stopped at the first backslash, so a model writing
# ``\textbf{Answer:} \textbf{YES}`` produced no match at all: the occurrence
# vanished, the list filled with None and the score came out 0/5. A different
# rendering of the same correct answer must not read as a collapse.
#
# Public on purpose: annotate.py has to rewrite exactly the occurrences this
# module reads, since it pairs them by position with the scoring results. Two
# copies of the pattern drift apart on the first edit, and the annotated
# document then shows one question the expected answer of another.
ANSWER_LINE_PATTERN = re.compile(r"\\textbf\{(?:Answer|Risposta):\}([^\n]*)")

# LaTeX control sequences have to go before the answer is looked for, or the
# command name gets read as the answer.
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
_WORD = re.compile(r"[A-Za-z]+")

# Plausible variants a model may produce despite the instructions: Italian
# tokens, accents, leftover punctuation. All normalise to YES or NO.
_TRUE_TOKENS = {"SI", "VERO", "TRUE", "YES", "Y"}
_FALSE_TOKENS = {"NO", "FALSO", "FALSE", "N"}


# The report is in English, so YES/NO is the canonical form. Italian tokens
# stay accepted, which keeps the reports produced before the switch scoreable
# without regenerating them.
ANSWER_TRUE = "YES"
ANSWER_FALSE = "NO"


def _to_ascii(text: str) -> str:
    """Reduce to ASCII, so that an accented answer matches its plain form."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


_TRUE_NORMALIZED = {_to_ascii(t).upper() for t in _TRUE_TOKENS}
_FALSE_NORMALIZED = {_to_ascii(t).upper() for t in _FALSE_TOKENS}


def _normalize_answer(raw: str) -> str | None:
    """Isolate the answer from the text following the label.

    Takes the first word left after stripping LaTeX commands. The answer comes
    first, so a comment trailing it, as in ``YES (the direct effect is
    large)``, does not get in the way. Returns None when that word is not a
    recognised token, which is the unreadable case.
    """
    words = _WORD.findall(_LATEX_COMMAND.sub(" ", _to_ascii(raw)))
    if not words:
        return None
    first = words[0].upper()
    if first in _TRUE_NORMALIZED:
        return ANSWER_TRUE
    if first in _FALSE_NORMALIZED:
        return ANSWER_FALSE
    return None


def parse_recap_answers(latex_text: str) -> list[str | None]:
    """Extract the answers in the order they appear in the document.

    Each element is YES, NO, or None when the text was unreadable or the
    answer is missing altogether, whether from an unfilled placeholder or from
    a model that added or dropped a question.
    """
    matches = ANSWER_LINE_PATTERN.findall(latex_text)
    answers = [_normalize_answer(m) for m in matches]
    # Pad or truncate to N_QUESTIONS: a model that gets the document
    # structure wrong must not crash the scorer, but the mismatch has to
    # remain visible.
    if len(answers) < N_QUESTIONS:
        answers = answers + [None] * (N_QUESTIONS - len(answers))
    return answers[:N_QUESTIONS]


###############################################################################
# Scoring
###############################################################################

@dataclass
class QuestionResult:
    index: int
    llm_answer: str | None
    ground_truth: str
    correct: bool


@dataclass
class ScoreReport:
    results: list[QuestionResult] = field(default_factory=list)
    n_correct: int = 0
    n_total: int = 0
    n_unparseable: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "score_pct": f"{self.score * 100:.1f}%",
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "n_unparseable": self.n_unparseable,
            "questions": [
                {
                    "index": r.index,
                    "llm_answer": r.llm_answer,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                }
                for r in self.results
            ],
        }


def score_report(latex_text: str, effects: dict[str, float]) -> ScoreReport:
    """Compare the answers read from the report with the expected ones.

    An unreadable answer always counts as wrong: a report that ignores the
    required format must not score full marks merely because nobody could
    read it.
    """
    llm_answers = parse_recap_answers(latex_text)
    ground_truths = [ANSWER_TRUE if rule(effects) else ANSWER_FALSE for rule in GROUND_TRUTH_RULES]

    report = ScoreReport(n_total=N_QUESTIONS)
    # strict=True: both lists must hold N_QUESTIONS items. A loose zip
    # would silently stop at the shorter one and score fewer questions
    # than exist.
    for i, (llm_ans, gt) in enumerate(zip(llm_answers, ground_truths, strict=True), start=1):
        correct = llm_ans == gt
        report.results.append(QuestionResult(i, llm_ans, gt, correct))
        if llm_ans is None:
            report.n_unparseable += 1
        if correct:
            report.n_correct += 1

    report.score = report.n_correct / report.n_total if report.n_total else 0.0
    return report


###############################################################################
# CLI
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a generated LaTeX report and score it against the "
        "expected answers derived from the exact FairMind values."
    )
    parser.add_argument("latex_path", help="Path to the generated .tex file")
    parser.add_argument(
        "--effects",
        required=True,
        help="Path to a JSON holding the keys TV, TE, SE, DE, IE and IE_reverse",
    )
    args = parser.parse_args()

    latex_text = open(args.latex_path, encoding="utf-8").read()
    effects = json.load(open(args.effects, encoding="utf-8"))

    report = score_report(latex_text, effects)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
