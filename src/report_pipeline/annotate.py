"""Annotazione del report generato con le risposte attese.

Il template che l'LLM riceve contiene solo i segnaposto delle sue risposte:
le risposte attese NON possono comparirvi, altrimenti il modello le
leggerebbe e le ricopierebbe, e il punteggio non misurerebbe piu' nulla.

Questo modulo interviene percio' a valle, sul documento gia' prodotto e
gia' valutato: riscrive la sezione "Recap Questions" affiancando a ogni
risposta del modello quella attesa e l'esito del confronto, e aggiunge una
riga di riepilogo con il punteggio. Il file originale resta disponibile
come evidenza di cio' che il modello ha effettivamente scritto.
"""

from __future__ import annotations

import re

from .validator import ScoreReport

# Cattura la riga "\textbf{Risposta:} XX" lasciando fuori l'eventuale coda,
# cosi' da poterla sostituire per intero mantenendo il testo della domanda.
_ANSWER_LINE = re.compile(r"\\textbf\{(?:Answer|Risposta):\}\s*([^\s\\}]+)")

_MISSING = "--"


def _verdict(correct: bool, llm_answer: str | None) -> str:
    if llm_answer is None:
        return "unreadable"
    return "correct" if correct else "wrong"


def annotate_recap_answers(latex_text: str, score: ScoreReport) -> str:
    """Affianca a ogni risposta dell'LLM quella attesa e l'esito.

    Sostituisce in ordine le occorrenze di ``\\textbf{Risposta:} X`` con
    ``\\textbf{LLM:} X \\quad \\textbf{Attesa:} Y \\quad (esito)``. L'ordine
    delle occorrenze nel documento e' lo stesso usato dal validator per
    estrarle, quindi l'accoppiamento con ``score.results`` e' garantito.

    Se il documento contiene piu' risposte di quante ne siano state valutate
    le eccedenti restano invariate, cosi' da non alterare silenziosamente un
    documento malformato.
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
    """Aggiunge dopo l'elenco una riga con il punteggio complessivo.

    La riga viene inserita subito prima di ``\\end{document}``: e' il solo
    punto di ancoraggio garantito, dal momento che la struttura interna del
    documento dipende da cio' che il modello ha prodotto.
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
