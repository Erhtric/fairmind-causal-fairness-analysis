"""Chiamata all'LLM per la generazione del report LaTeX.

``src.llm.call_llm()`` non e' riutilizzabile qui perche' fa il parsing di un
blocco JSON dalla risposta e solleva un errore se non lo trova: quella e' la
firma giusta per il benchmark degli effetti (2_3), dove il modello deve
restituire TV/TE/DE/IE, ma in questa pipeline l'output atteso e' il sorgente
LaTeX completo. Questo modulo fa la stessa chiamata (stesso endpoint, stessa
configurazione, ``temperature=0``) ma restituisce il testo grezzo.
"""

from __future__ import annotations

import re
import time

from openai import OpenAI

from ..llm import LLM_CONFIGS

# Un LLM tende a incapsulare l'output in un blocco markdown nonostante le
# istruzioni contrarie: lo togliamo qui invece di penalizzare il report.
_FENCE_PATTERN = re.compile(r"^\s*```(?:latex|tex)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)

_PLACEHOLDER_PATTERN = re.compile(r"<<([A-Z0-9_]+)>>")


def strip_markdown_fences(text: str) -> str:
    match = _FENCE_PATTERN.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def extract_latex_document(text: str) -> str:
    """Isola il documento da ``\\documentclass`` a ``\\end{document}``.

    Serve a scartare eventuale testo di accompagnamento che il modello
    aggiunge prima o dopo il sorgente, senza toccare il documento stesso.
    """
    cleaned = strip_markdown_fences(text)
    start = cleaned.find(r"\documentclass")
    end = cleaned.rfind(r"\end{document}")
    if start == -1 or end == -1:
        return cleaned
    return cleaned[start : end + len(r"\end{document}")]


def find_unfilled_placeholders(latex_text: str) -> list[str]:
    """Segnaposto ``<<NOME>>`` rimasti nel documento.

    Una lista non vuota significa che il modello non ha completato il
    template: il report va considerato malformato.
    """
    return _PLACEHOLDER_PATTERN.findall(latex_text)


def call_llm_report(
    system_prompt: str,
    user_prompt: str,
    config: dict | None = None,
    max_tokens: int = 4096,
    cache_prompt: bool = False,
) -> tuple[str, dict, float]:
    """Invia i due prompt e restituisce ``(latex, usage, elapsed_seconds)``.

    Il LaTeX e' gia' ripulito da fence markdown e da eventuale testo fuori
    dal documento; NON viene validato qui (se ne occupa il validator).
    """
    if config is None:
        config = LLM_CONFIGS[0]

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        # Come in src.llm.call_llm: llama.cpp assegna la richiesta allo slot
        # con il prefisso comune piu' lungo e ne riusa la KV cache, per cui
        # richieste identiche possono divergere anche a temperatura nulla.
        # Disattivata perche' un report non riproducibile non e' valutabile.
        extra_body={"cache_prompt": cache_prompt},
    )
    elapsed = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason,
    }

    latex = extract_latex_document(response.choices[0].message.content)
    return latex, usage, elapsed
