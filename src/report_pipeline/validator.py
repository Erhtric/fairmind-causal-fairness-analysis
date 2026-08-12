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

from dataclasses import dataclass, field
import argparse
import json
import re
import unicodedata


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
# INVERSA. Con la forma diretta la risposta a Q2 si ribalta, silenziosamente:
# su Adult IE_reverse = -0.046333 (mitiga TE) e IE = +0.016869 (non mitiga).
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
    """Q2: l'effetto indiretto mitiga il totale? Serve segno opposto a TE, e non trascurabile."""
    ie, te = decomposition_ie(effects), effects["TE"]
    if abs(ie) < EPS_IE:
        return False
    return (ie * te) < 0


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

_ANSWER_PATTERN = re.compile(r"\\textbf\{(?:Answer|Risposta):\}\s*([^\s\\}]+)")

# Normalizza varianti plausibili che un LLM potrebbe produrre nonostante le
# istruzioni (accenti, VERO/FALSO, inglese, punteggiatura residua) verso
# esattamente "SI" o "NO".
_TRUE_TOKENS = {"SI", "SÌ", "VERO", "TRUE", "YES", "Y"}
_FALSE_TOKENS = {"NO", "FALSO", "FALSE", "N"}


# Il report e' in inglese, quindi la forma canonica e' YES/NO. I token
# italiani restano accettati: i report prodotti prima del passaggio
# all'inglese continuano cosi' a essere valutabili senza rigenerarli.
ANSWER_TRUE = "YES"
ANSWER_FALSE = "NO"


def _normalize_answer(raw: str) -> str | None:
    cleaned = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z]", "", cleaned).upper()
    if cleaned in {t.encode("ascii", "ignore").decode("ascii").upper() for t in _TRUE_TOKENS}:
        return ANSWER_TRUE
    if cleaned in {t.encode("ascii", "ignore").decode("ascii").upper() for t in _FALSE_TOKENS}:
        return ANSWER_FALSE
    return None


def parse_recap_answers(latex_text: str) -> list[str | None]:
    """Estrae le risposte SI/NO nell'ordine in cui compaiono nel documento.

    Ogni elemento e' "SI", "NO", o None se il testo trovato non era
    interpretabile o se una risposta manca del tutto (placeholder non
    riempito, o LLM che ha aggiunto/rimosso una domanda).
    """
    matches = _ANSWER_PATTERN.findall(latex_text)
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
    for i, (llm_ans, gt) in enumerate(zip(llm_answers, ground_truths), start=1):
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
