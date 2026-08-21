"""Parser e scorer per i report LaTeX generati dall'LLM (punto 4 del feedback
del docente).

Il report contiene una sezione "Recap Questions" con 5 domande a risposta
booleana (SI/NO), riempite dall'LLM in base alla propria interpretazione
qualitativa. Questo script:

1. Fa il parsing delle 5 risposte dal file LaTeX (regex, nessun bisogno di
   compilare il documento).
2. Calcola una ground truth deterministica per le stesse 5 domande, a partire
   SOLO dai valori numerici esatti di FairMind (le soglie sono costanti
   modificabili in cima al file).
3. Confronta le due liste e restituisce uno score di consistenza.

Uso:
    python -m src.report_pipeline.validator report.tex --effects effects.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field

###############################################################################
# Soglie quantitative per la ground truth — modificare qui.
###############################################################################

THRESH_DE = 0.05        # |DE| oltre questa soglia => discriminazione diretta
EPS_IE = 0.005          # |IE| sotto questa soglia => canale trascurabile, non "mitiga"
THRESH_TV = 0.05        # |TV| oltre questa soglia => effetto totale praticamente rilevante
THRESH_SE_REL = 0.10    # |SE| / |TV| oltre questa soglia => componente spuria sostanziale

# Il paper definisce due effetti indiretti distinti, che NON sono l'uno
# l'opposto dell'altro (v. 2_3_benchmark_thor.ipynb). Le Recap Questions Q2 e
# Q5 parlano della decomposizione TE = DE - IE, quindi richiedono la forma
# INVERSA. Con la forma diretta la risposta a Q2 si ribalta, silenziosamente.
# La chiave e' esplicita, e la sua assenza e' un errore, non un default.
IE_DECOMPOSITION_KEY = "IE_reverse"


def decomposition_ie(effects: dict[str, float]) -> float:
    """L'effetto indiretto nella forma che entra nella decomposizione.

    Solleva un errore se la chiave manca, invece di ripiegare su "IE": un
    ripiego silenzioso produrrebbe una ground truth sbagliata su Q2 senza
    alcun sintomo osservabile nel punteggio.
    """
    if IE_DECOMPOSITION_KEY not in effects:
        raise KeyError(
            f"'{IE_DECOMPOSITION_KEY}' assente dagli effetti. Le regole di "
            "ground truth riguardano la decomposizione TE = DE - IE e "
            "richiedono la forma inversa dell'effetto indiretto; passare "
            "la forma diretta cambierebbe la risposta a Q2."
        )
    return effects[IE_DECOMPOSITION_KEY]


###############################################################################
# Ground truth: una funzione per domanda, stesso ordine del template.
###############################################################################

def gt_q1_direct_discrimination(effects: dict[str, float]) -> bool:
    """Q1: esiste discriminazione diretta? DE sufficientemente grande in modulo."""
    return abs(effects["DE"]) > THRESH_DE


def gt_q2_ie_mitigates_te(effects: dict[str, float]) -> bool:
    """Q2: l'effetto indiretto mitiga il totale, o lo amplifica?

    Confermato dal relatore (Antonucci, 20/08/2026) via email: posto
    IE_prof := -IE_{x1,x0}, vale TE = DE + IE_prof (Prop. 2 in forma
    additiva). Se DE e IE_prof CONCORDANO in segno, l'effetto mediato
    AMPLIFICA il totale (|TE| > |DE|); se DISCORDANO, lo ATTENUA
    (|TE| < |DE|).

    decomposition_ie() restituisce IE_{x1,x0}, cioe' IE_prof cambiato di
    segno. "DE concorde con IE_prof" equivale percio' a "DE discorde con
    decomposition_ie()", e viceversa: la mitigazione (segni DISCORDI in
    IE_prof) corrisponde a decomposition_ie() CONCORDE con DE, cioe' un
    prodotto POSITIVO.

    Verificato con un calcolo numerico diretto sui valori di Adult, non
    solo algebricamente, perche' una prima stesura di questa docstring
    aveva il verso scambiato: decomposition_ie = -0.046333, DE = +0.138404,
    prodotto negativo, quindi la funzione restituisce False (non mitiga,
    cioe' amplifica) -- coerente con la lettura del relatore. Regression
    test in tests/test_report_pipeline_validator.py.
    """
    ie, de = decomposition_ie(effects), effects["DE"]
    if abs(ie) < EPS_IE:
        return False
    return (ie * de) > 0


def gt_q3_tv_practically_relevant(effects: dict[str, float]) -> bool:
    """Q3: il TV supera la soglia di rilevanza pratica?"""
    return abs(effects["TV"]) > THRESH_TV


def gt_q4_se_substantial(effects: dict[str, float]) -> bool:
    """Q4: la componente spuria pesa abbastanza rispetto al totale osservato?"""
    tv, se = effects["TV"], effects["SE"]
    if abs(tv) < 1e-9:
        return abs(se) > 0
    return abs(se) / abs(tv) > THRESH_SE_REL


def gt_q5_de_dominant_over_ie(effects: dict[str, float]) -> bool:
    """Q5: la componente diretta domina su quella indiretta?"""
    return abs(effects["DE"]) > abs(decomposition_ie(effects))


# Ordine = ordine delle domande nel template (src/report_pipeline/template.tex).
GROUND_TRUTH_RULES = [
    gt_q1_direct_discrimination,
    gt_q2_ie_mitigates_te,
    gt_q3_tv_practically_relevant,
    gt_q4_se_substantial,
    gt_q5_de_dominant_over_ie,
]

N_QUESTIONS = len(GROUND_TRUTH_RULES)


###############################################################################
# Parsing delle risposte SI/NO dal LaTeX generato dall'LLM.
###############################################################################

# Cattura tutto cio' che segue l'etichetta fino a fine riga, e lascia a
# _normalize_answer il compito di isolare la risposta.
#
# La versione precedente, ``\s*([^\s\\}]+)``, si fermava al primo backslash:
# un modello che scrive ``\textbf{Answer:} \textbf{YES}`` non produceva alcun
# match, l'occorrenza spariva, la lista veniva riempita di None e il punteggio
# risultava 0/5. Un rendering diverso della stessa risposta corretta non deve
# poter essere scambiato per un collasso del modello.
#
# Pubblico di proposito: annotate.py deve riscrivere ESATTAMENTE le stesse
# occorrenze che questo modulo legge, perche' le accoppia per posizione con i
# risultati dello scoring. Due copie della stessa regex si disallineano alla
# prima modifica, e il documento annotato mostrerebbe su una domanda la
# risposta attesa di un'altra.
ANSWER_LINE_PATTERN = re.compile(r"\\textbf\{(?:Answer|Risposta):\}([^\n]*)")

# Sequenze di controllo LaTeX (``\textbf``, ``\emph``, ...): vanno rimosse
# prima di cercare la risposta, altrimenti il nome del comando verrebbe letto
# come se fosse la risposta stessa.
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
_WORD = re.compile(r"[A-Za-z]+")

# Normalizza varianti plausibili che un LLM potrebbe produrre nonostante le
# istruzioni (accenti, VERO/FALSO, inglese, punteggiatura residua) verso
# esattamente "YES" o "NO".
_TRUE_TOKENS = {"SI", "SÌ", "VERO", "TRUE", "YES", "Y"}
_FALSE_TOKENS = {"NO", "FALSO", "FALSE", "N"}


# Il report e' in inglese, quindi la forma canonica e' YES/NO. I token
# italiani restano accettati: i report prodotti prima del passaggio
# all'inglese continuano cosi' a essere valutabili senza rigenerarli.
ANSWER_TRUE = "YES"
ANSWER_FALSE = "NO"


def _to_ascii(text: str) -> str:
    """Riduce ad ASCII, cosi' che "SÌ" e "SI" siano lo stesso token."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


_TRUE_NORMALIZED = {_to_ascii(t).upper() for t in _TRUE_TOKENS}
_FALSE_NORMALIZED = {_to_ascii(t).upper() for t in _FALSE_TOKENS}


def _normalize_answer(raw: str) -> str | None:
    """Isola la risposta dal testo che segue l'etichetta.

    Prende la **prima** parola rimasta dopo aver tolto i comandi LaTeX: la
    risposta viene per prima, e un eventuale commento che la segue
    (``YES (the direct effect is large)``) non deve impedirne la lettura.
    Restituisce None se quella parola non e' un token riconosciuto, che e' il
    caso in cui la risposta non e' interpretabile.
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
    """Estrae le risposte SI/NO nell'ordine in cui compaiono nel documento.

    Ogni elemento e' "SI", "NO", o None se il testo trovato non era
    interpretabile o se una risposta manca del tutto (placeholder non
    riempito, o LLM che ha aggiunto/rimosso una domanda).
    """
    matches = ANSWER_LINE_PATTERN.findall(latex_text)
    answers = [_normalize_answer(m) for m in matches]
    # Padding/troncamento a N_QUESTIONS: un LLM che sbaglia la struttura del
    # documento (aggiunge o toglie una domanda) non deve far crashare lo
    # scorer, ma va segnalato come mismatch strutturale.
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
    """Confronta le risposte estratte dal report con la ground truth.

    Una risposta non parsabile (None) conta sempre come errore: un report
    che non rispetta il formato richiesto non puo' ottenere punteggio pieno
    solo perche' non e' stato possibile leggerlo.
    """
    llm_answers = parse_recap_answers(latex_text)
    ground_truths = [ANSWER_TRUE if rule(effects) else ANSWER_FALSE for rule in GROUND_TRUTH_RULES]

    report = ScoreReport(n_total=N_QUESTIONS)
    # strict=True: le due liste devono avere entrambe N_QUESTIONS elementi.
    # Uno zip non stretto troncherebbe in silenzio sulla piu' corta, e il
    # punteggio verrebbe calcolato su meno domande di quante ne esistono.
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
        description="Parsa e valuta un report LaTeX generato dall'LLM contro "
        "la ground truth deterministica calcolata sui valori esatti di FairMind."
    )
    parser.add_argument("latex_path", help="Percorso del file .tex generato dall'LLM")
    parser.add_argument(
        "--effects",
        required=True,
        help="Percorso di un JSON con le chiavi TV, TE, SE, DE, IE (i valori esatti di FairMind)",
    )
    args = parser.parse_args()

    latex_text = open(args.latex_path, encoding="utf-8").read()
    effects = json.load(open(args.effects, encoding="utf-8"))

    report = score_report(latex_text, effects)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
