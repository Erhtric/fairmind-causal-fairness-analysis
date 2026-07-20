# Interpretazione della probabilità da parte di un LLM

Esperimento richiesto dal docente, indipendente dalla pipeline FairMind: verifica se un LLM interpreta la probabilità in modo frequentista o bayesiano, tramite una conversazione a tre turni.

## Come eseguirlo

**Con Qwen su Thor** (stesso server usato nel resto della tesi):

```bash
export LLAMA_HOST=<nodo del server llama_server_gpu attivo, es. gnode04>
export LLAMA_PORT=8080
jupyter nbconvert --to notebook --execute \
    --output analyze_llm_probability_interpretation_output.ipynb \
    analyze_llm_probability_interpretation.ipynb
```

**In locale, con OpenAI**: apri il notebook e sostituisci la cella di setup del client con `client = OpenAI()` (legge `OPENAI_API_KEY` dall'ambiente) e `MODEL_NAME` con il modello desiderato (es. `gpt-4o-mini`).

## Output

Ogni esecuzione salva la conversazione completa (prompt, risposte, token usati) in `results/probability_interpretation_<timestamp>.json`.

## Nota

Questo notebook non è stato ancora eseguito da me: non ho accesso diretto a un endpoint LLM (né al server Thor via SSH, né a una chiave OpenAI) da questo ambiente. Va lanciato manualmente con uno dei due metodi sopra.
