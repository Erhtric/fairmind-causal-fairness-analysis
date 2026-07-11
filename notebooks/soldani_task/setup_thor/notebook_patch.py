"""
Patch per il notebook 2_1_benchmark_thor.ipynb.

Sostituisce le celle di connessione al LLM, leggendo dinamicamente
l'host dal job Slurm (variabile d'ambiente LLAMA_HOST) o dal file di log.

Modalita' d'uso:
    1. Sostituisci la cella `client = OpenAI(...)` con questo codice.
    2. Sostituisci la funzione `call_gpt` con quella qui sotto.
"""

import os
import re
import time
from pathlib import Path
from openai import OpenAI
import json

# --- 1. Determina hostname e porta ---
LLAMA_HOST = os.environ.get("LLAMA_HOST")
LLAMA_PORT = os.environ.get("LLAMA_PORT", "8080")

if LLAMA_HOST is None:
    # Fallback: cerca l'ultimo file llama_server_<jobid>.out
    out_files = sorted(Path(".").glob("llama_server_*.out"), reverse=True)
    if out_files:
        content = out_files[0].read_text()
        match = re.search(r"Job in esecuzione su: (\S+)", content)
        if match:
            LLAMA_HOST = match.group(1)
            print(f"Letto host da {out_files[0].name}: {LLAMA_HOST}")
        else:
            LLAMA_HOST = "localhost"
            print("Host non trovato, fallback a localhost")
    else:
        LLAMA_HOST = "localhost"
        print("Nessun file .out trovato, fallback a localhost")

# --- 2. Connessione ---
client = OpenAI(
    base_url=f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1",
    api_key="not-needed",
)
MODEL_NAME = "qwen2.5-7b-instruct"

print(f"Connesso a http://{LLAMA_HOST}:{LLAMA_PORT}/v1")


# --- 3. Sostituisci la funzione call_gpt con questa ---
def call_gpt(prompt: str, model: str = MODEL_NAME, max_retries: int = 2) -> tuple[dict, dict, float]:
    """
    Chiama l'LLM con retry e validazione dell'output.
    
    Returns:
        (effetti, usage, elapsed)
    
    Raises:
        ValueError: se dopo max_retries l'output non e' valido
    """
    last_error = None
    for attempt in range(1 + max_retries):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            elapsed = time.perf_counter() - start

            usage = {
                "input_tokens":     response.usage.prompt_tokens,
                "output_tokens":    response.usage.completion_tokens,
                "reasoning_tokens": None,
                "total_tokens":     response.usage.total_tokens,
            }

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            effects = json.loads(raw)

            # Validazione base
            required = {"TV", "TE", "SE", "DE", "IE"}
            missing = required - set(effects.keys())
            if missing:
                raise ValueError(f"Chiavi mancanti: {missing}")

            all_zero = all(abs(float(effects.get(k, 1))) < 1e-9 for k in required)
            if all_zero:
                print(f"Attenzione: tutti gli effetti sono zero (tentativo {attempt+1})")

            return effects, usage, elapsed

        except (json.JSONDecodeError, ValueError, Exception) as e:
            last_error = e
            print(f"Tentativo {attempt+1} fallito: {e}")
            if attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"Riprovo tra {wait}s...")
                time.sleep(wait)

    raise ValueError(f"LLM non ha prodotto output valido dopo {max_retries+1} tentativi. Ultimo errore: {last_error}")
